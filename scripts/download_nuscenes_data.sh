#!/bin/bash

function get_download_url() {
  data_option=${1:-"v1.0-trainval01_blobs.tgz"}
  common_download_url="https://s3.ap-southeast-1.amazonaws.com/asia.data.nuscenes.org/public/v1.0/${data_option}?AWSAccessKeyId=ASIA6RIK4RRMHMMFDMOD&Signature=UI%2FiGcVbsNGpsF9t%2Bda%2FJUQ2OzU%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEPb%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQDLb%2BuUb%2FjfOvH8BB28XBBvMi2ddzcgwW68%2FBZTS1CnvQIgKyWx5IeXKnavXu7dILK7kpLd2p%2BNjrkxMzrJ%2BuCwZlUq9AIIPxAEGgw5OTkxMzk2MDk2ODgiDI5fs%2BSi17LzbcXDxCrRAnndnDj38Gh4IqyAuTp%2BoXJ5FdQyBQK%2Fc1wVy7cqK86ceUVP79ciTYtoIFocV5c1GvaSITrIwchBRBKNpmK4x4%2Bxkk4sD5dMDLCUOdUgtFnjRMFMDYQVbT7subMttqy5OsgU3i%2Bv6XI%2BkKMYtynDJkbQe29%2FaIRh7ogB6hQOgyoqePPMIHIWlLu9f8%2FV166DqjvlJW9rq7HAcbjmWdhLWy%2FsB2%2Be6DDLhLCFtIvVhSgxHkK41EaOLAoEOvCik3F14M92DAe%2BiMKepVtD7GPRS5MU9%2FxPeySKJgh2gjP9Xai6b3afjETrcEABvtdeoeMS7fOCV2Isgq7LoTHKZzZrd2ilry1M%2FXFi13M2Pb714%2F9a6ux%2BS1MjkAghiuuhy02O6fL2apOpPGk%2BlHChZDqLMZdPU%2FPhi5g6PZ7XLTMvFZottONbaq%2Bq%2BBK0sY7GHPqqGYgwzPHGqgY6ngE5NyM%2F8U0rKE%2BgtkXCKljgP9FkWkWUXRZuZngfKQWRwhNc4VXpbqom7kKvpifVGoRIjwKi32taqxP9IsmRjgh09pPG7WFwbTrLchIGMbS7DGT%2FANUv8943HXarYenMecO1qi9ViCg1bqKJVGM5lCkFJbx7Q5F7t4VGBekqxOAwuIbGsTkP8PapLkOat%2FsWIK5V5anjIYM6jkV3iIIsDg%3D%3D&Expires=1700289913"
  echo $common_download_url
}

if [ -z "$@" ]; then
  echo "#download "
  echo "https://blog.csdn.net/jin15203846657/article/details/125739016"
fi

for i in {1..10}
do
  version="0$i"
  if [ $i -ge 10 ]; then
    version="$i"
  fi
  data_option="v1.0-trainval${version}_blobs.tar"
  echo $data_option
  common_download_url=$(get_download_url $data_option)

  wget -c -O $data_option $common_download_url

  # echo ${common_download_url:0:100}
done

data_option="v1.0-trainval_meta.tgz" wget -c -O $data_option $common_download_url
data_option="v1.0-test_blobs.tgz" wget -c -O $data_option $common_download_url
data_option="v1.0-test_meta.tgz" wget -c -O $data_option $common_download_url
