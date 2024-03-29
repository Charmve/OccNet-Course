#!/bin/bash

sudo apt-get remove --purge -y *nvidia*450*
sudo apt-get remove --purge -y *nvidia*450*:i386
sudo apt-get remove --purge -y *nvidia*450*:amd64
sudo apt-get remove --purge -y *nvidia*455*
sudo apt-get remove --purge -y *nvidia*455*:i386
sudo apt-get remove --purge -y *nvidia*455*:amd64
sudo apt-get remove --purge -y *nvidia*460*
sudo apt-get remove --purge -y *nvidia*460*:i386
sudo apt-get remove --purge -y *nvidia*460*:amd64

sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:graphics-drivers/ppa

sudo apt-get update

ubuntu-drivers devices
sudo apt-get install -y nvidia-driver-470

sudo reboot
