/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
#include <pcl/segmentation/extract_clusters.h>

#include "cuda_runtime.h"
#include "lib/cudaCluster.h"

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

void testCUDA(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::ratio<1, 1000>> time_span =
     std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);

  cudaStream_t stream = NULL;
  cudaStreamCreate (&stream);

  std::cout << "-------------- cudaExtractCluster -----------"<< std::endl;
  /*add cuda cluster*/
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudNew(new pcl::PointCloud<pcl::PointXYZ>);
  cloudNew = cloud;
  float *inputEC = NULL;
  unsigned int sizeEC = cloudNew->size();
  cudaMallocManaged(&inputEC, sizeof(float) * 4 * sizeEC, cudaMemAttachHost);
  cudaStreamAttachMemAsync (stream, inputEC);
  cudaMemcpyAsync(inputEC, cloudNew->points.data(), sizeof(float) * 4 * sizeEC, cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);

  float *outputEC = NULL;
  cudaMallocManaged(&outputEC, sizeof(float) * 4 * sizeEC, cudaMemAttachHost);
  cudaStreamAttachMemAsync (stream, outputEC);
  cudaMemcpyAsync(outputEC, cloudNew->points.data(), sizeof(float) * 4 * sizeEC, cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);

  unsigned int *indexEC = NULL;
  cudaMallocManaged(&indexEC, sizeof(float) * 4 * sizeEC, cudaMemAttachHost);
  cudaStreamAttachMemAsync (stream, indexEC);
  cudaMemsetAsync(indexEC, 0, sizeof(float) * 4 * sizeEC, stream);
  cudaStreamSynchronize(stream);

  extractClusterParam_t ecp;
  ecp.minClusterSize = 100;
  ecp.maxClusterSize = 2500000;
  ecp.voxelX = 0.05;
  ecp.voxelY = 0.05;
  ecp.voxelZ = 0.05;
  ecp.countThreshold = 20;
  cudaExtractCluster cudaec(stream);
  cudaec.set(ecp);

  t1 = std::chrono::steady_clock::now();
  cudaec.extract(inputEC, sizeEC, outputEC, indexEC);
  cudaStreamSynchronize(stream);
  t2 = std::chrono::steady_clock::now();
  time_span = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);
  std::cout << "CUDA extract by Time: " << time_span.count() << " ms."<< std::endl;

  pcl::PCDWriter writer;
  int j = 0;
  for (int i = 1; i <= indexEC[0]; i++)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);

    cloud_cluster->width  = indexEC[i];
    cloud_cluster->height = 1;
    cloud_cluster->points.resize (cloud_cluster->width * cloud_cluster->height);
    cloud_cluster->is_dense = true;

    unsigned int outoff = 0;
    for (int w = 1; w < i; w++)
    {
      if (i>1) {
        outoff += indexEC[w];
      }
    }

    for (std::size_t k = 0; k < indexEC[i]; ++k)
    {
      cloud_cluster->points[k].x = outputEC[(outoff+k)*4+0];
      cloud_cluster->points[k].y = outputEC[(outoff+k)*4+1];
      cloud_cluster->points[k].z = outputEC[(outoff+k)*4+2];
    }

    std::cout << "PointCloud representing the Cluster: " << cloud_cluster->size () << " data points." << std::endl;
    std::stringstream ss;
    j++;
    ss << "cuda_cloud_cluster_" << j << ".pcd";
    writer.write<pcl::PointXYZ> (ss.str (), *cloud_cluster, false); //*
  }

  cudaFree(inputEC);
  cudaFree(outputEC);
  cudaFree(indexEC);
  /*end*/
}

void testPCL(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::ratio<1, 1000>> time_span =
     std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ>), cloud_f (new pcl::PointCloud<pcl::PointXYZ>);
  cloud_f = cloud;
  // cluster
  // Creating the KdTree object for the search method of the extraction
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);

  t1 = std::chrono::steady_clock::now();
  tree->setInputCloud (cloud);
  t2 = std::chrono::steady_clock::now();
  time_span = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);
  std::cout << "PCL(CPU) cluster kd-tree by Time: " << time_span.count() << " ms."<< std::endl;

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance (0.02); // 2cm
  ec.setMinClusterSize (50);
  ec.setMaxClusterSize (2500000);
  ec.setSearchMethod (tree);
  ec.setInputCloud (cloud_f);

  t1 = std::chrono::steady_clock::now();
  ec.extract (cluster_indices);
  t2 = std::chrono::steady_clock::now();
  time_span = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);
  std::cout << "PCL(CPU) cluster extracted by Time: " << time_span.count() << " ms."<< std::endl;

  std::cout << "PointCloud cluster_indices: " << cluster_indices.size () << "." << std::endl;

  pcl::PCDWriter writer;
  int j = 1;
  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
    for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
      cloud_cluster->push_back ((*cloud_f)[*pit]); //*
    cloud_cluster->width = cloud_cluster->size ();
    cloud_cluster->height = 1;
    cloud_cluster->is_dense = true;

    std::cout << "PointCloud representing the Cluster: " << cloud_cluster->size () << " data points." << std::endl;
    std::stringstream ss;
    ss << "cloud_cluster_" << j << ".pcd";
    writer.write<pcl::PointXYZ> (ss.str (), *cloud_cluster, false); //*
    j++;
  }
}

int main(int argc, const char **argv)
{
  std::string file = "./sample.pcd";
  if(argc > 1) file = (argv[1]);

  Getinfo();

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudSrc(new pcl::PointCloud<pcl::PointXYZ>);

  if (pcl::io::loadPCDFile<pcl::PointXYZ>(file.c_str(), *cloudSrc )== -1)
  {
    std::cout << "Error:can not open the file: "<< file.c_str() << std::endl;
    return(-1);
  }

  std::cout << "-------------- test CUDA lib -----------"<< std::endl;
  testCUDA(cloudSrc);

  std::cout << "\n-------------- test PCL lib -----------"<< std::endl;
  testPCL(cloudSrc);

  return 0;
}
