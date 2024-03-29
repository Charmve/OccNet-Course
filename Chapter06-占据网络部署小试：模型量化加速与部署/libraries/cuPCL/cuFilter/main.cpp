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
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include "cuda_runtime.h"
#include "lib/cudaFilter.h"

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

  cloudDst->width  = nCount;
  cloudDst->height = 1;
  cloudDst->resize (cloudDst->width * cloudDst->height);

  float *outputData = (float *)cloudDst->points.data();

  memset(outputData,0,sizeof(float)*4*nCount);

  std::cout << "\n------------checking CUDA ---------------- "<< std::endl;
  std::cout << "CUDA Loaded "
      << cloudSrc->width*cloudSrc->height
      << " data points from PCD file with the following fields: "
      << pcl::getFieldsList (*cloudSrc)
      << std::endl;

  float *input = NULL;
  cudaMallocManaged(&input, sizeof(float) * 4 * nCount, cudaMemAttachHost);
  cudaStreamAttachMemAsync (stream, input );
  cudaMemcpyAsync(input, inputData, sizeof(float) * 4 * nCount, cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);

  float *output = NULL;
  cudaMallocManaged(&output, sizeof(float) * 4 * nCount, cudaMemAttachHost);
  cudaStreamAttachMemAsync (stream, output );
  cudaStreamSynchronize(stream);

  cudaFilter filterTest(stream);
  FilterParam_t setP;
  FilterType_t type;

{
  unsigned int countLeft = 0;
  std::cout << "\n------------checking CUDA PassThrough ---------------- "<< std::endl;

  memset(outputData,0,sizeof(float)*4*nCount);

  FilterType_t type = PASSTHROUGH;

  setP.type = type;
  setP.dim = 0;
  setP.upFilterLimits = 0.5;
  setP.downFilterLimits = -0.5;
  setP.limitsNegative = false;
  filterTest.set(setP);

  cudaDeviceSynchronize();
  t1 = std::chrono::steady_clock::now();
  filterTest.filter(output, &countLeft, input, nCount);
  checkCudaErrors(cudaMemcpyAsync(outputData, output, sizeof(float) * 4 * countLeft, cudaMemcpyDeviceToHost, stream));
  checkCudaErrors(cudaDeviceSynchronize());
  t2 = std::chrono::steady_clock::now();
  time_span = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);
  std::cout << "CUDA PassThrough by Time: " << time_span.count() << " ms." << std::endl;
  std::cout << "CUDA PassThrough before filtering: " << nCount << std::endl;
  std::cout << "CUDA PassThrough after filtering: " << countLeft << std::endl;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudNew(new pcl::PointCloud<pcl::PointXYZ>);
  cloudNew->width = countLeft;
  cloudNew->height = 1;
  cloudNew->points.resize (cloudNew->width * cloudNew->height);

  int check = 0;
  for (std::size_t i = 0; i < cloudNew->size(); ++i)
  {
      cloudNew->points[i].x = output[i*4+0];
      cloudNew->points[i].y = output[i*4+1];
      cloudNew->points[i].z = output[i*4+2];
  }
  pcl::io::savePCDFileASCII ("after-cuda-PassThrough.pcd", *cloudNew);
}

{
  unsigned int countLeft = 0;
  std::cout << "\n------------checking CUDA VoxelGrid---------------- "<< std::endl;

  memset(outputData,0,sizeof(float)*4*nCount);

  type = VOXELGRID;

  setP.type = type;
  setP.voxelX = 1;
  setP.voxelY = 1;
  setP.voxelZ = 1;

  filterTest.set(setP);
  int status = 0;
  cudaDeviceSynchronize();
  t1 = std::chrono::steady_clock::now();
  status = filterTest.filter(output, &countLeft, input, nCount);
  cudaDeviceSynchronize();
  t2 = std::chrono::steady_clock::now();

  if (status != 0)
    return;
  time_span = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);
  std::cout << "CUDA VoxelGrid by Time: " << time_span.count() << " ms."<< std::endl;
  std::cout << "CUDA VoxelGrid before filtering: " << nCount << std::endl;
  std::cout << "CUDA VoxelGrid after filtering: " << countLeft << std::endl;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudNew(new pcl::PointCloud<pcl::PointXYZ>);
  cloudNew->width = countLeft;
  cloudNew->height = 1;
  cloudNew->points.resize (cloudNew->width * cloudNew->height);

  int check = 0;
  for (std::size_t i = 0; i < cloudNew->size(); ++i)
  {
      cloudNew->points[i].x = output[i*4+0];
      cloudNew->points[i].y = output[i*4+1];
      cloudNew->points[i].z = output[i*4+2];
  }
  pcl::io::savePCDFileASCII ("after-cuda-VoxelGrid.pcd", *cloudNew);
}

  cudaFree(input);
  cudaFree(output);
  cudaStreamDestroy(stream);
}

void testPCL(pcl::PointCloud<pcl::PointXYZ>::Ptr cloudSrc,
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudDst)
{
  std::cout << "\n\n------------checking PCL ---------------- "<< std::endl;
  std::cout << "PCL(CPU) Loaded "
      << cloudSrc->width*cloudSrc->height
      << " data points from PCD file with the following fields: "
      << pcl::getFieldsList (*cloudSrc)
      << std::endl;
  
  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::ratio<1, 1000>> time_span =
     std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);

  int nCount = cloudSrc->width * cloudSrc->height;
  float *outputData = (float *)cloudDst->points.data();
{
  std::cout << "\n------------checking PCL(CPU) PassThrough ---------------- "<< std::endl;

  memset(outputData,0,sizeof(float)*4*nCount);

  // Create the filtering object
  pcl::PassThrough<pcl::PointXYZ> pass;
  pass.setInputCloud (cloudSrc);
  pass.setFilterFieldName ("x");
  pass.setFilterLimits (-0.5, 0.5);
  pass.setFilterLimitsNegative (false);

  t1 = std::chrono::steady_clock::now();
  pass.filter (*cloudDst);
  t2 = std::chrono::steady_clock::now();

  time_span = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);
  std::cout << "PCL(CPU) PassThrough by Time: " << time_span.count() << " ms."<< std::endl;

  std::cout << "PointCloud before filtering: " << cloudSrc->width * cloudSrc->height 
   << " data points (" << pcl::getFieldsList (*cloudSrc) << ")." << std::endl;
  std::cout << "PointCloud after filtering: " << cloudDst->width * cloudDst->height 
     << " data points (" << pcl::getFieldsList (*cloudDst) << ")." << std::endl;
  pcl::io::savePCDFileASCII ("after-pcl-PassThrough.pcd", *cloudDst);
}

{
  std::cout << "\n------------checking PCL VoxelGrid---------------- "<< std::endl;

  memset(outputData,0,sizeof(float)*4*nCount);

  t1 = std::chrono::steady_clock::now();

  // Create the filtering object
  pcl::VoxelGrid<pcl::PointXYZ> sor;
  sor.setInputCloud (cloudSrc);
  sor.setLeafSize (1, 1, 1);
  sor.filter (*cloudDst);

  t2 = std::chrono::steady_clock::now();
  time_span = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);
  std::cout << "PCL VoxelGrid by Time: " << time_span.count() << " ms."<< std::endl;
  std::cout << "PointCloud before filtering: " << cloudSrc->width * cloudSrc->height 
   << " data points (" << pcl::getFieldsList (*cloudSrc) << ")." << std::endl;
  std::cout << "PointCloud after filtering: " << cloudDst->width * cloudDst->height 
     << " data points (" << pcl::getFieldsList (*cloudDst) << ")." << std::endl;

  pcl::io::savePCDFileASCII ("after-pcl-VoxelGrid.pcd", *cloudDst);
}

}

int main(int argc, const char **argv)
{
  std::string file = "./sample.pcd";
  if(argc > 1) file = (argv[1]);

  Getinfo();

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudSrc(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudDst(new pcl::PointCloud<pcl::PointXYZ>);

  if (pcl::io::loadPCDFile<pcl::PointXYZ>(file.c_str(), *cloudSrc )== -1)
  {
    std::cout << "Error:can not open the file: "<< file.c_str() << std::endl;
    return(-1);
  }

  testCUDA(cloudSrc, cloudDst);
  testPCL(cloudSrc, cloudDst);

  return 0;
}
