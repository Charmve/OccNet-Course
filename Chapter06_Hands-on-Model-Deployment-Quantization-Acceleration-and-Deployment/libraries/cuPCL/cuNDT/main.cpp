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

#include <pcl/registration/ndt.h>

#include "cuda_runtime.h"
#include "lib/cudaNDT.h"

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

/******************************************************
Function: cloud points transform
inputs: ConP input cloud points (szie 3*N),
        transformation_matrix for two cloud points (szie 4*4)
output: NewP = transformation_matrix * ConP (size 3*N)
********************************************************/
Eigen::MatrixXf Transform(const Eigen::MatrixXf P, const Eigen::MatrixXf Transmatrix, int PCountN)
{
  int N = PCountN;
  Eigen::MatrixXf R = Transmatrix.block(0, 0, 3, 3);
  Eigen::VectorXf T = Transmatrix.block(0, 3, 3, 1);
  Eigen::MatrixXf ConP = P.block(0, 0, 3, N);
  //std::cout << "Matrix"  <<  R.cols() <<" " << ConP.rows()<< std::endl;
  Eigen::MatrixXf NewP(4,N);
  Eigen::MatrixXf NewP3N = (R*ConP).colwise() + T;
  for (int i = 0; i <N; i++) {
    NewP(0,i) = NewP3N(0,i);
    NewP(1,i) = NewP3N(1,i);
    NewP(2,i) = NewP3N(2,i);
    NewP(3,i) = 0.0f;
  }

  return NewP;
}

void print4x4Matrix(const Eigen::Matrix4f & matrix)
{
  printf("Rotation matrix :\n");
  printf("    | %f %f %f | \n", matrix(0, 0), matrix(0, 1), matrix(0, 2));
  printf("R = | %f %f %f | \n", matrix(1, 0), matrix(1, 1), matrix(1, 2));
  printf("    | %f %f %f | \n", matrix(2, 0), matrix(2, 1), matrix(2, 2));
  printf("Translation vector :\n");
  printf("t = < %f, %f, %f >\n\n", matrix(0, 3), matrix(1, 3), matrix(2, 3));
}

double calculateFitneeScore(pcl::PointCloud<pcl::PointXYZ>::Ptr P,
        pcl::PointCloud<pcl::PointXYZ>::Ptr Q,
        Eigen::Matrix4f transformation_matrix)
{
  double fitness_score = 0.0;
  pcl::PointCloud<pcl::PointXYZ> input_transformed;
  pcl::transformPointCloud (*P, input_transformed, transformation_matrix);

  pcl::search::KdTree<pcl::PointXYZ> tree_;
  std::vector<int> nn_indices (1);
  std::vector<float> nn_dists (1);

  tree_.setInputCloud(Q);
  int nr = 0;
  for (std::size_t i = 0; i < input_transformed.points.size (); ++i)
  {
    // Find its nearest neighbor in the target
    tree_.nearestKSearch (input_transformed.points[i], 1, nn_indices, nn_dists);
    if (nn_dists[0] <=  std::numeric_limits<double>::max ())
    {
      // Add to the fitness score
      fitness_score += nn_dists[0];
      nr++;
    }
  }
  if (nr > 0)
    return (fitness_score / nr);
  return (std::numeric_limits<double>::max ());
}

void testcudaNDT(pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_in,
        pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_out,
        Eigen::Matrix4f &transformation_matrix)
{
  {
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudALL (new pcl::PointCloud<pcl::PointXYZRGB>(pcl_cloud_in->size() + pcl_cloud_out->size(),1));
  for (int i = 0; i < pcl_cloud_in->size(); i++)
  {
    pcl::PointXYZRGB pointin;
    pointin.x = (*pcl_cloud_in)[i].x;
    pointin.y = (*pcl_cloud_in)[i].y;
    pointin.z = (*pcl_cloud_in)[i].z;
    pointin.r = 255;
    pointin.g = 0;
    pointin.b = 0;
    (*cloudALL)[i] = pointin;
  }
  for (int i = 0; i < pcl_cloud_out->size(); i++)
  {
    pcl::PointXYZRGB pointout;
    pointout.x = (*pcl_cloud_out)[i].x;
    pointout.y = (*pcl_cloud_out)[i].y;
    pointout.z = (*pcl_cloud_out)[i].z;
    pointout.r = 0;
    pointout.g = 255;
    pointout.b = 255;
    (*cloudALL)[i+pcl_cloud_in->size()] = pointout;
  }

  pcl::io::savePCDFile<pcl::PointXYZRGB> ("basic.pcd", *cloudALL);
  }

  int nP = pcl_cloud_in->size();
  int nQ = pcl_cloud_out->size();
  float *nPdata = (float *)pcl_cloud_in->points.data();
  float *nQdata = (float *)pcl_cloud_out->points.data();
  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::ratio<1, 1000>> time_span =
     std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);

  Eigen::Matrix4f matrix_trans = Eigen::Matrix4f::Identity();
  //std::cout << "matrix_trans native value "<< std::endl;
  //print4x4Matrix(matrix_trans);

  Eigen::AngleAxisf init_rotation (0.6931, Eigen::Vector3f::UnitZ ());
  Eigen::Translation3f init_translation (1.79387, 0.720047, 0);
  Eigen::Matrix4f init_guess = (init_translation * init_rotation).matrix ();
  init_guess = matrix_trans;

  std::cout << "------------checking CUDA NDT(GPU)---------------- "<< std::endl;
  /************************************************/
  cudaStream_t stream = NULL;
  cudaStreamCreate ( &stream );

  float *PUVM = NULL;
  cudaMallocManaged(&PUVM, sizeof(float) * 4 * nP, cudaMemAttachHost);
  cudaStreamAttachMemAsync (stream, PUVM );
  cudaMemcpyAsync(PUVM, nPdata, sizeof(float) * 4 * nP, cudaMemcpyHostToDevice, stream);

  float *QUVM = NULL;
  cudaMallocManaged(&QUVM, sizeof(float) * 4 * nQ, cudaMemAttachHost);
  cudaStreamAttachMemAsync (stream, QUVM );
  cudaMemcpyAsync(QUVM, nQdata, sizeof(float) * 4 * nQ, cudaMemcpyHostToDevice, stream);

  float *guess = NULL;
  cudaMallocManaged(&guess, sizeof(float) * 4 * 4, cudaMemAttachHost);
  cudaStreamAttachMemAsync (stream, guess);
  cudaMemcpyAsync(guess, init_guess.data(), sizeof(float) * 4 * 4, cudaMemcpyHostToDevice, stream);

  float *cudaMatrix = NULL;
  cudaMallocManaged(&cudaMatrix, sizeof(float) * 4 * 4, cudaMemAttachHost);
  cudaStreamAttachMemAsync (stream, cudaMatrix);
  cudaMemsetAsync(cudaMatrix, 0, sizeof(float) * 4 * 4, stream);

  cudaStreamSynchronize(stream);

  cudaNDT ndtTest(nP, nQ, stream);

  ndtTest.setInputSource ((void*)PUVM);
  ndtTest.setInputTarget ((void*)QUVM);
  ndtTest.setResolution (1.0);
  ndtTest.setMaximumIterations (35);
  ndtTest.setTransformationEpsilon (0.01);
  ndtTest.setStepSize (0.1);

  t1 = std::chrono::steady_clock::now();
  ndtTest.ndt((float*)PUVM, nP, (float*)QUVM, nQ, guess, cudaMatrix, stream);
  t2 = std::chrono::steady_clock::now();
  time_span = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);
  std::cout << "CUDA NDT by Time: " << time_span.count() << " ms."<< std::endl;
  cudaStreamDestroy(stream);
  /************************************************/
  memcpy(matrix_trans.data(), cudaMatrix, sizeof(float)*4*4);
  transformation_matrix = matrix_trans;
  std::cout << "CUDA NDT fitness_score: " << calculateFitneeScore (pcl_cloud_in, pcl_cloud_out, transformation_matrix) << std::endl;
  print4x4Matrix (matrix_trans);

  cudaFree(PUVM);
  cudaFree(QUVM);
  cudaFree(cudaMatrix);
  cudaFree(guess);

  auto cloudSrc = pcl_cloud_in;
  auto cloudDst = pcl_cloud_out;
  pcl::PointCloud<pcl::PointXYZ> input_transformed;
  pcl::transformPointCloud (*cloudSrc, input_transformed, transformation_matrix);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudSrcRGB (new pcl::PointCloud<pcl::PointXYZRGB>(cloudSrc->size(),1));
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudDstRGB (new pcl::PointCloud<pcl::PointXYZRGB>(cloudDst->size(),1));
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudALL (new pcl::PointCloud<pcl::PointXYZRGB>(cloudSrc->size() + cloudDst->size(),1));
  // Fill in the CloudIn data
  for (int i = 0; i < cloudSrc->size(); i++)
  {
    pcl::PointXYZRGB &pointin = (*cloudSrcRGB)[i];
    pointin.x = (input_transformed)[i].x;
    pointin.y = (input_transformed)[i].y;
    pointin.z = (input_transformed)[i].z;
    pointin.r = 255;
    pointin.g = 0;
    pointin.b = 0;
    (*cloudALL)[i] = pointin;
  }
  for (int i = 0; i < cloudDst->size(); i++)
  {
    pcl::PointXYZRGB &pointout = (*cloudDstRGB)[i];
    pointout.x = (*cloudDst)[i].x;
    pointout.y = (*cloudDst)[i].y;
    pointout.z = (*cloudDst)[i].z;
    pointout.r = 0;
    pointout.g = 255;
    pointout.b = 255;
    (*cloudALL)[i+cloudSrc->size()] = pointout;
  }

  pcl::io::savePCDFile<pcl::PointXYZRGB> ("cuda.pcd", *cloudALL);
}

void testPCLNDT(pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_in,
        pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_out
        )
{
  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::ratio<1, 1000>> time_span =
     std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);

  Eigen::Matrix4f matrix_trans = Eigen::Matrix4f::Identity();

  std::cout << "------------checking PCL NDT(CPU)---------------- "<< std::endl;
  int pCount = pcl_cloud_in->size();

  pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
  ndt.setTransformationEpsilon (0.01);
  ndt.setStepSize (0.1);
  ndt.setResolution (1.0);
  ndt.setMaximumIterations (35);
  ndt.setInputSource (pcl_cloud_in);
  ndt.setInputTarget (pcl_cloud_out);

  // Set initial alignment estimate found using robot odometry.
  Eigen::AngleAxisf init_rotation (0.6931, Eigen::Vector3f::UnitZ ());
  Eigen::Translation3f init_translation (1.79387, 0.720047, 0);
  Eigen::Matrix4f init_guess = (init_translation * init_rotation).matrix ();
  init_guess = matrix_trans;

  // Calculating required rigid transform to align the input cloud to the target cloud.
  pcl::PointCloud<pcl::PointXYZ>::Ptr transformedP (new pcl::PointCloud<pcl::PointXYZ>(pCount, 1));
  t1 = std::chrono::steady_clock::now();
  ndt.align (*transformedP, init_guess);
  t2 = std::chrono::steady_clock::now();

  time_span = std::chrono::duration_cast<std::chrono::duration<double, std::ratio<1, 1000>>>(t2 - t1);
  std::cout << "PCL align Time: " << time_span.count() << " ms."<< std::endl;
  std::cout << "Normal Distributions Transform has converged: " << ndt.hasConverged ()
            << " score: " << ndt.getFitnessScore () << std::endl;

  auto transformation_matrix =  ndt.getFinalTransformation ();
  print4x4Matrix (transformation_matrix);

  auto cloudSrc = pcl_cloud_in;
  auto cloudDst = pcl_cloud_out;

  pcl::PointCloud<pcl::PointXYZ> input_transformed;
  pcl::transformPointCloud (*cloudSrc, input_transformed, transformation_matrix);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudSrcRGB (new pcl::PointCloud<pcl::PointXYZRGB>(cloudSrc->size(),1));
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudDstRGB (new pcl::PointCloud<pcl::PointXYZRGB>(cloudDst->size(),1));
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudALL (new pcl::PointCloud<pcl::PointXYZRGB>(cloudSrc->size() + cloudDst->size(),1));
  // Fill in the CloudIn data
  for (int i = 0; i < cloudSrc->size(); i++)
  {
    pcl::PointXYZRGB &pointin = (*cloudSrcRGB)[i];
    pointin.x = (input_transformed)[i].x;
    pointin.y = (input_transformed)[i].y;
    pointin.z = (input_transformed)[i].z;
    pointin.r = 255;
    pointin.g = 0;
    pointin.b = 0;
    (*cloudALL)[i] = pointin;
  }
  for (int i = 0; i < cloudDst->size(); i++)
  {
    pcl::PointXYZRGB &pointout = (*cloudDstRGB)[i];
    pointout.x = (*cloudDst)[i].x;
    pointout.y = (*cloudDst)[i].y;
    pointout.z = (*cloudDst)[i].z;
    pointout.r = 0;
    pointout.g = 255;
    pointout.b = 255;
    (*cloudALL)[i+cloudSrc->size()] = pointout;
  }

  pcl::io::savePCDFile<pcl::PointXYZRGB> ("ndt.pcd", *cloudALL);
}

int main(int argc, const char **argv)
{
  Getinfo();

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudSrc(new pcl::PointCloud<pcl::PointXYZ>);
  if (pcl::io::loadPCDFile<pcl::PointXYZ>("test_P.pcd", *cloudSrc )== -1)
  {
    PCL_ERROR("Couldn't read file test_pcd.pcd\n");
    return(-1);
  }
  std::cout << "Loaded " << cloudSrc->width*cloudSrc->height
      << " data points for P with the following fields: "
      << pcl::getFieldsList (*cloudSrc)
      << std::endl;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudDst(new pcl::PointCloud<pcl::PointXYZ>);
  if (pcl::io::loadPCDFile<pcl::PointXYZ>("test_Q.pcd", *cloudDst )== -1)
  {
    PCL_ERROR("Couldn't read file test_pcd.pcd\n");
    return(-1);
  }
  std::cout << "Loaded " << cloudDst->width*cloudDst->height
      << " data points for Q with the following fields: "
      << pcl::getFieldsList (*cloudSrc)
      << std::endl;

  // Defining a rotation matrix and translation vector
  Eigen::Matrix4f transformation_matrix = Eigen::Matrix4f::Identity();

  // A rotation matrix (see https://en.wikipedia.org/wiki/Rotation_matrix)
  double theta = M_PI / 8;  // The angle of rotation in radians
  transformation_matrix(0, 0) = cos(theta);
  transformation_matrix(0, 1) = -sin(theta);
  transformation_matrix(1, 0) = sin(theta);
  transformation_matrix(1, 1) = cos(theta);
  // A translation on Z axis
  transformation_matrix(2, 3) = 0.2;
  transformation_matrix(1, 3) = 0;

  // Display in terminal the transformation matrix
  std::cout << "Target rigid transformation : cloud_P -> cloud_Q" << std::endl;
  print4x4Matrix(transformation_matrix);

  //////////////////// filter ///////////////
  int totalPoints = cloudSrc->size();
  int usePoints = cloudSrc->size() * 1;

  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_in (new pcl::PointCloud<pcl::PointXYZ>(usePoints,1));
  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_out (new pcl::PointCloud<pcl::PointXYZ>(cloudDst->size(),1));
  // Fill in the CloudIn data
  for (int i = 0; i < pcl_cloud_in->size(); i++)
  {
    pcl::PointXYZ &pointin = (*pcl_cloud_in)[i];
    pointin.x = (*cloudSrc)[i* (totalPoints/usePoints)].x;
    pointin.y = (*cloudSrc)[i* (totalPoints/usePoints)].y;
    pointin.z = (*cloudSrc)[i* (totalPoints/usePoints)].z;
  }
  for (int i = 0; i < pcl_cloud_out->size(); i++)
  {
    pcl::PointXYZ &pointout = (*pcl_cloud_out)[i];
    pointout.x = (*cloudDst)[i* (totalPoints/totalPoints)].x;
    pointout.y = (*cloudDst)[i* (totalPoints/totalPoints)].y;
    pointout.z = (*cloudDst)[i* (totalPoints/totalPoints)].z;
  }
  //////////////////// filter ///////////////
  testPCLNDT( pcl_cloud_in, pcl_cloud_out);

  testcudaNDT( pcl_cloud_in, pcl_cloud_out, transformation_matrix);


  return 0;
}
