# FlexRAN™ FEC SDK Modules

Copyright (C) 2019-2023 Intel Corporation

Introduction
============
FlexRAN™ FEC SDK Modules provides optimized libraries for LTE and for 5G NR Layer 1 workload acceleration. FlexRAN™ FEC SDK is intended for use with machines that support Intel® Advanced Vector Extensions 2 (Intel® AVX2) and Intel® Advanced Vector Extensions 512 (Intel® AVX-512) instruction sets.

This set of libraries supports Forward Error Correction (FEC), rate matching, and cyclic redundancy check (CRC) functions as specified in the 3rd Generation Partnership Project (3GPP) standards.

The kit includes the following resources:

APIs
Source code
Build and test environment parameters
The driver and build instructions for the Data Plane Development Kit (DPDK) Wireless Baseband device library (BBDEV) virtual Poll Mode Driver (PMD) that support these libraries is available at dpdk.org.

The API and implementation are subject to change without notice.

The software dependencies include:

Intel® One API Compiler (ICX)
CMake minimum version 2.8.12
Intel® Integrated Performance Primitives (IPP) 18.0
Intel® Math Kernel Library (Intel® MKL) 18.0
gtest Google Test 1.7.0 (required to run the verification and compute performance tests)

The information on how to get the community version of the Intel® ICX compiler is available at:
https://docs.o-ran-sc.org/projects/o-ran-sc-o-du-phy/en/latest/build_prerequisite.html#download-and-install-oneapi
for the instructions on how to download the compilers.

For the large binary files used for self-testing git lfs is used


Unit Tests
==========
For functional and performance testing google test is used. Please refer to the User Manual under the section running unit tests. To run all tests use ./test/phy/run_all_test.


Documentation
=============
The documentation for this Project is available in html format. Please unzip the file html.zip available in the doc/doxygen folder and open the index.html file
The Main Page contains the Revision History, Release Notes and links to the User manual and Programmers Guide and also from the Upper Tabs acces to the Data Structures and Files