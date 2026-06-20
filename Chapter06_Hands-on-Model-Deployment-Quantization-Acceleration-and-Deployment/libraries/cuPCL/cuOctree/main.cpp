/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#include <fstream>
#include <chrono>

#include <pcl/io/pcd_io.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <pcl/octree/octree_search.h> 
#include <pcl/point_types.h>

#include <iostream>
#include <vector>

#include "cuda_runtime.h"
#include "lib/cudaOctree.h"


void Getinfo(void)
{
    cudaDeviceProp prop;

    int count = 0;
    cudaGetDeviceCount(&count);
    printf("\nGPU has cuda devices: %d\n", count);
    for (int i = 0; i < count; ++i) {
        cudaGetDeviceProperties(&prop, i);
        printf("----device id: %d info----\n", i);
        printf("  GPU : %s \n", prop.name);
        printf("  Capbility: %d.%d\n", prop.major, prop.minor);
        printf("  Global memory: %luMB\n", prop.totalGlobalMem >> 20);
        printf("  Const memory: %luKB\n", prop.totalConstMem  >> 10);
        printf("  SM in a block: %luKB\n", prop.sharedMemPerBlock >> 10);
        printf("  warp size: %d\n", prop.warpSize);
        printf("  threads in a block: %d\n", prop.maxThreadsPerBlock);
        printf("  block dim: (%d,%d,%d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  grid dim: (%d,%d,%d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    }
    printf("\n");
}

void testCUDA(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudSrc,
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudDst)
{
  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::ratio<1, 1000>> time_span =
     std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);

  cudaStream_t stream = NULL;
  cudaStreamCreate ( &stream );

  unsigned int nCount = cloudSrc->width * cloudSrc->height;
  float *inputData = (float *)cloudSrc->points.data();

  unsigned int nDstCount = cloudDst->width * cloudDst->height;
  float *outputData = (float *)cloudDst->points.data();

  float *input = NULL;//points cloud source which be searched
  cudaMallocManaged(&input, sizeof(float) * 4 * nCount, cudaMemAttachHost);
  cudaStreamAttachMemAsync (stream, input);
  cudaMemcpyAsync(input, inputData, sizeof(float) * 4 * nCount, cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);

  float *output = NULL;// Dst is the targets points
  cudaMallocManaged(&output, sizeof(float) * 4 *nDstCount, cudaMemAttachHost);
  cudaStreamAttachMemAsync (stream, output);
  cudaMemsetAsync(output, 0, sizeof(unsigned int), stream);
  cudaMemcpyAsync(output, outputData, sizeof(float) * 4 * nDstCount, cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);

  float *search = NULL;//search point (one point)
  cudaMallocManaged(&search, sizeof(float) * 4, cudaMemAttachHost);
  cudaStreamAttachMemAsync (stream, search);
  cudaStreamSynchronize(stream);

  unsigned int *selectedCount = NULL;//count of points selected
  checkCudaErrors(cudaMallocManaged(&selectedCount, sizeof(unsigned int)*nDstCount, cudaMemAttachHost));
  checkCudaErrors(cudaStreamAttachMemAsync(stream, selectedCount) );
  checkCudaErrors(cudaMemsetAsync(selectedCount, 0, sizeof(unsigned int)*nDstCount, stream));

  int *index = NULL;//index selected by search
  cudaMallocManaged(&index, sizeof(int) * nCount, cudaMemAttachHost);
  cudaStreamAttachMemAsync (stream, index);
  cudaMemsetAsync(index, 0, sizeof(unsigned int), stream);
  cudaStreamSynchronize(stream);

  float *distance = NULL;//suqure distance between points selected by search
  cudaMallocManaged(&distance, sizeof(float) * nCount, cudaMemAttachHost);
  cudaStreamAttachMemAsync (stream, distance);
  cudaMemsetAsync(distance, 0, sizeof(unsigned int), stream);
  cudaStreamSynchronize(stream);

  float resolution = 0.03f;
  cudaTree treeTest(input, nCount, resolution, stream);

{
  cudaMemsetAsync(index, 0, sizeof(unsigned int), stream);
  cudaMemsetAsync(distance, 0xFF, sizeof(unsigned int), stream);
  cudaMemsetAsync(selectedCount, 0, sizeof(unsigned int), stream);
  cudaStreamSynchronize(stream);
  std::cout << "\n------------checking CUDA Approx nearest search---------------- "<< std::endl;

  int *pointIdxANSearch = index;
  float *pointANSquaredDistance = distance;
  int status = 0;
  *selectedCount = nDstCount;//how many points in DST(output)

  cudaDeviceSynchronize();
  t1 = std::chrono::steady_clock::now();

  status = treeTest.approxNearestSearch(output, pointIdxANSearch, pointANSquaredDistance, selectedCount);

  cudaDeviceSynchronize();
  t2 = std::chrono::steady_clock::now();

  if (status != 0) return;

  time_span = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);
  std::cout << "CUDA costs : " << time_span.count() << " ms."<< std::endl;

  double distanceAll = 0.0;
  for(int i = 0; i < *selectedCount; i ++) {
    distanceAll += ( *( ((unsigned int*)pointANSquaredDistance) + i) )/1e9;
  }

  std::cout << "Point distance AVG: " << distanceAll/(*selectedCount) << std::endl;
  cudaStreamSynchronize(stream);
}

{
  cudaMemsetAsync(index, 0, sizeof(unsigned int), stream);
  cudaMemsetAsync(distance, 0, sizeof(unsigned int), stream);
  cudaMemsetAsync(selectedCount, 0, sizeof(unsigned int), stream);
  cudaStreamSynchronize(stream);

  std::cout << "\n------------checking CUDA radiusSearch---------------- "<< std::endl;

  *((float4*)search) = {-0.87431729, 2.2932131, -0.65834892, 0.0};
  float radius = 10.06;
  int *pointIdxRadiusSearch = index;
  float *pointRadiusSquaredDistance = distance;
  int status = 0;

  cudaDeviceSynchronize();
  t1 = std::chrono::steady_clock::now();
  status = treeTest.radiusSearch(search, radius,
      pointIdxRadiusSearch, pointRadiusSquaredDistance, selectedCount);
  cudaDeviceSynchronize();
  t2 = std::chrono::steady_clock::now();

  if (status != 0) return;

  time_span = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);
  std::cout << "CUDA costs: " << time_span.count() << " ms."<< std::endl;
  std::cout << "Points selected: " << *selectedCount << std::endl;

}

  cudaFree(search);
  cudaFree(index);
  cudaFree(input);
  cudaFree(output);
  cudaFree(distance);
  cudaFree(selectedCount);
  cudaStreamDestroy(stream);
}

void testPCL(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudSrc,
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudDst)
{
  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::ratio<1, 1000>> time_span =
     std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);

  int nCount = cloudSrc->width * cloudSrc->height;
  float *outputData = (float *)cloudDst->points.data();

{//octree
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  cloud->resize(cloudSrc->width * cloudSrc->height);
  cloud->width = cloudSrc->width;
  cloud->height = cloudSrc->height;
  for (size_t i = 0; i < cloud->points.size(); ++i) {
	  cloud->points[i].x = cloudSrc->points[i].x;
	  cloud->points[i].y = cloudSrc->points[i].y;
	  cloud->points[i].z = cloudSrc->points[i].z;
  }

  for (size_t i = 0; i < cloud->points.size(); ++i) {
	  cloud->points[i].r = 0;
	  cloud->points[i].g = 0;
	  cloud->points[i].b = 0;
  }

  float resolution = 0.03f;
  pcl::octree::OctreePointCloudSearch<pcl::PointXYZRGB> octree(resolution);

  t1 = std::chrono::steady_clock::now();
  octree.setInputCloud(cloud);
  octree.addPointsFromInputCloud();
  t2 = std::chrono::steady_clock::now();
  time_span = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);
  std::cout << "\n------------checking OC-Tree creating ---------------- "<< std::endl;
  std::cout << "PCL(CPU) create oc-tree by Time: " << time_span.count() << " ms."<< std::endl;

  float time = 0.0;

  double distanceALL = 0.0;
  std::cout << "\n------------checking PCL(CPU)  Approx nearest search ---------------- "<< std::endl;
  for (int i =0; i < cloudSrc->width * cloudSrc->height; i++) {
	pcl::PointXYZRGB searchPoint1 = cloud->points[3000];
  searchPoint1.x = cloudDst->points[i].x;
  searchPoint1.y = cloudDst->points[i].y;
  searchPoint1.z = cloudDst->points[i].z;
  searchPoint1.r = 0;
  searchPoint1.g = 0;
  searchPoint1.b = 0;
	std::vector<int> pointIdxVec;
  int result_index;
  float sqr_distance;

  t1 = std::chrono::steady_clock::now();
  octree.approxNearestSearch(searchPoint1, result_index, sqr_distance);
  t2 = std::chrono::steady_clock::now();
  time_span = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);
  time += time_span.count();
  distanceALL += sqr_distance;
  }
  std::cout << "PCL(CPU) costs: " << time << " ms."<< std::endl;
  std::cout << "Point distance AVG: " << distanceALL/(cloudSrc->width * cloudSrc->height) << std::endl;

{
	pcl::PointXYZRGB searchPoint3 = cloud->points[3000];
  searchPoint3.x = -0.87431729;
  searchPoint3.y = 2.2932131;
  searchPoint3.z = -0.65834892;
  searchPoint3.r = 0;
  searchPoint3.g = 0;
  searchPoint3.b = 0;

	std::vector<int> pointIdxRadiusSearch;
	std::vector<float> pointRadiusSquaredDistance;
	float radius = 10.06;
  std::cout << "\n------------checking PCL(CPU) radiusSearch ---------------- "<< std::endl;
  t1 = std::chrono::steady_clock::now();
  auto result = octree.radiusSearch(searchPoint3, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance); 
  t2 = std::chrono::steady_clock::now();
  time_span = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);

  std::cout << "PCL(CPU) costs by Time: " << time_span.count() << " ms."<< std::endl;
	std::cout << "Points selected: " << pointIdxRadiusSearch.size() << std::endl;
}

}//octree


}

int main(int argc, const char **argv)
{
  std::string file = "./test_P.pcd";
  if(argc > 1) file = (argv[1]);

  std::string file1 = "./test_Q.pcd";
  if(argc > 1) file1 = (argv[2]);

  Getinfo();

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudSrc(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudDst(new pcl::PointCloud<pcl::PointXYZ>);

  if (pcl::io::loadPCDFile<pcl::PointXYZ>(file.c_str(), *cloudSrc )== -1)
  {
    std::cout << "Error:can not open the file: "<< file.c_str() << std::endl;
    return(-1);
  }

  if (pcl::io::loadPCDFile<pcl::PointXYZ>(file1.c_str(), *cloudDst )== -1)
  {
    std::cout << "Error:can not open the file: "<< file1.c_str() << std::endl;
    return(-1);
  }
  testCUDA(cloudSrc, cloudDst);
  testPCL(cloudSrc, cloudDst);

  return 0;
}
