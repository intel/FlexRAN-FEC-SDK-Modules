/**********************************************************************
*
*
*  Copyright [2019 - 2023] [Intel Corporation]
* 
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  
*  You may obtain a copy of the License at
*  
*     http://www.apache.org/licenses/LICENSE-2.0 
*  
*  Unless required by applicable law or agreed to in writing, software 
*  distributed under the License is distributed on an "AS IS" BASIS, 
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and 
*  limitations under the License. 
*  
*  SPDX-License-Identifier: Apache-2.0 
*  
* 
*
**********************************************************************/

#include "LdpcDecoder.hpp"
#include "Tables.hpp"

/// The positions in the 46 x 68 basegraph matrix of the circulants
/// Each row of text represents a row of the basegraph
/// BG1
const uint8_t k_bg1ColumnPositions[SimdLdpc::k_maxCirculants] = {
  0, 1, 2, 3, 5, 6, 9, 10, 11, 12, 13, 15, 16, 18, 19, 20, 21, 22, 23,
  0, 2, 3, 4, 5, 7, 8, 9, 11, 12, 14, 15, 16, 17, 19, 21, 22, 23, 24,
  0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 17, 18, 19, 20, 24, 25,
  0, 1, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 25,
  0, 1, 26,
  0, 1, 3, 12, 16, 21, 22, 27,
  0, 6, 10, 11, 13, 17, 18, 20, 28,
  0, 1, 4, 7, 8, 14, 29,
  0, 1, 3, 12, 16, 19, 21, 22, 24, 30,
  0, 1, 10, 11, 13, 17, 18, 20, 31,
  1, 2, 4, 7, 8, 14, 32,
  0, 1, 12, 16, 21, 22, 23, 33,
  0, 1, 10, 11, 13, 18, 34,
  0, 3, 7, 20, 23, 35,
  0, 12, 15, 16, 17, 21, 36,
  0, 1, 10, 13, 18, 25, 37,
  1, 3, 11, 20, 22, 38,
  0, 14, 16, 17, 21, 39,
  1, 12, 13, 18, 19, 40,
  0, 1, 7, 8, 10, 41,
  0, 3, 9, 11, 22, 42,
  1, 5, 16, 20, 21, 43,
  0, 12, 13, 17, 44,
  1, 2, 10, 18, 45,
  0, 3, 4, 11, 22, 46,
  1, 6, 7, 14, 47,
  0, 2, 4, 15, 48,
  1, 6, 8, 49,
  0, 4, 19, 21, 50,
  1, 14, 18, 25, 51,
  0, 10, 13, 24, 52,
  1, 7, 22, 25, 53,
  0, 12, 14, 24, 54,
  1, 2, 11, 21, 55,
  0, 7, 15, 17, 56,
  1, 6, 12, 22, 57,
  0, 14, 15, 18, 58,
  1, 13, 23, 59,
  0, 9, 10, 12, 60,
  1, 3, 7, 19, 61,
  0, 8, 17, 62,
  1, 3, 9, 18, 63,
  0, 4, 24, 64,
  1, 16, 18, 25, 65,
  0, 7, 9, 22, 66,
  1, 6, 10, 67
};

/// The row-weights of BG1
const uint8_t k_bg1RowWeights[SimdLdpc::k_maxRows] = {
  19, 19, 19, 19, 3, 8, 9, 7, 10, 9, 7, 8, 7, 6, 7, 7, 6, 6, 6, 6, 6, 6, 5, 5, 6, 5, 5, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 5, 5, 4, 5, 4, 5, 5, 4
};
const int16_t k_bg1RowWeightsCumulative[SimdLdpc::k_maxRows] = {
  19, 38, 57, 76, 79, 87, 96, 103, 113, 122, 129, 137, 144, 150, 157, 164, 170, 176, 182, 188, 194, 200, 205, 210, 216, 221, 226, 230, 235, 240, 245, 250, 255, 260, 265, 270, 275, 279, 284, 289, 293, 298, 302, 307, 312, 316
};

/// The circulant values of BG1 for a=2
/// This will be used for all values of Z in the set {2,4,8,16,32,64,128,256}
/// This is a*2^n with a = 2;
const int16_t k_bg1a2[SimdLdpc::k_maxCirculants] = {
  250, 69, 226, 159, 100, 10, 59, 229, 110, 191, 9, 195, 23, 190, 35, 239, 31, 1, 0,
  2, 239, 117, 124, 71, 222, 104, 173, 220, 102, 109, 132, 142, 155, 255, 28, 0, 0, 0,
  106, 111, 185, 63, 117, 93, 229, 177, 95, 39, 142, 225, 225, 245, 205, 251, 117, 0, 0,
  121, 89, 84, 20, 150, 131, 243, 136, 86, 246, 219, 211, 240, 76, 244, 144, 12, 1, 0,
  157, 102, 0,
  205, 236, 194, 231, 28, 123, 115, 0,
  183, 22, 28, 67, 244, 11, 157, 211, 0,
  220, 44, 159, 31, 167, 104, 0,
  112, 4, 7, 211, 102, 164, 109, 241, 90, 0,
  103, 182, 109, 21, 142, 14, 61, 216, 0,
  98, 149, 167, 160, 49, 58, 0,
  77, 41, 83, 182, 78, 252, 22, 0,
  160, 42, 21, 32, 234, 7, 0,
  177, 248, 151, 185, 62, 0,
  206, 55, 206, 127, 16, 229, 0,
  40, 96, 65, 63, 75, 179, 0,
  64, 49, 49, 51, 154, 0,
  7, 164, 59, 1, 144, 0,
  42, 233, 8, 155, 147, 0,
  60, 73, 72, 127, 224, 0,
  151, 186, 217, 47, 160, 0,
  249, 121, 109, 131, 171, 0,
  64, 142, 188, 158, 0,
  156, 147, 170, 152, 0,
  112, 86, 236, 116, 222, 0,
  23, 136, 116, 182, 0,
  195, 243, 215, 61, 0,
  25, 104, 194, 0,
  128, 165, 181, 63, 0,
  86, 236, 84, 6, 0,
  216, 73, 120, 9, 0,
  95, 177, 172, 61, 0,
  221, 112, 199, 121, 0,
  2, 187, 41, 211, 0,
  127, 167, 164, 159, 0,
  161, 197, 207, 103, 0,
  37, 105, 51, 120, 0,
  198, 220, 122, 0,
  167, 151, 157, 163, 0,
  173, 139, 149, 0, 0,
  157, 137, 149, 0,
  167, 173, 139, 151, 0,
  149, 157, 137, 0,
  151, 163, 173, 139, 0,
  139, 157, 163, 173, 0,
  149, 151, 167, 0
};

/// The circulant values of BG1 for a=3
/// This will be used for all values of Z in the set {3, 6, 12, 24, 48, 96, 192, 384}
/// This is a*2^n with a = 3;
const int16_t k_bg1a3[SimdLdpc::k_maxCirculants] = {
  307, 19, 50, 369, 181, 216, 317, 288, 109, 17, 357, 215, 106, 242, 180, 330, 346, 1, 0,
  76, 76, 73, 288, 144, 331, 331, 178, 295, 342, 217, 99, 354, 114, 331, 112, 0, 0, 0,
  205, 250, 328, 332, 256, 161, 267, 160, 63, 129, 200, 88, 53, 131, 240, 205, 13, 0, 0,
  276, 87, 0, 275, 199, 153, 56, 132, 305, 231, 341, 212, 304, 300, 271, 39, 357, 1, 0,
  332, 181, 0,
  195, 14, 115, 166, 241, 51, 157, 0,
  278, 257, 1, 351, 92, 253, 18, 225, 0,
  9, 62, 316, 333, 290, 114, 0,
  307, 179, 165, 18, 39, 224, 368, 67, 170, 0,
  366, 232, 321, 133, 57, 303, 63, 82, 0,
  101, 339, 274, 111, 383, 354, 0,
  48, 102, 8, 47, 188, 334, 115, 0,
  77, 186, 174, 232, 50, 74, 0,
  313, 177, 266, 115, 370, 0,
  142, 248, 137, 89, 347, 12, 0,
  241, 2, 210, 318, 55, 269, 0,
  13, 338, 57, 289, 57, 0,
  260, 303, 81, 358, 375, 0,
  130, 163, 280, 132, 4, 0,
  145, 213, 344, 242, 197, 0,
  187, 206, 264, 341, 59, 0,
  205, 102, 328, 213, 97, 0,
  30, 11, 233, 22, 0,
  24, 89, 61, 27, 0,
  298, 158, 235, 339, 234, 0,
  72, 17, 383, 312, 0,
  71, 81, 76, 136, 0,
  194, 194, 101, 0,
  222, 19, 244, 274, 0,
  252, 5, 147, 78, 0,
  159, 229, 260, 90, 0,
  100, 215, 258, 256, 0,
  102, 201, 175, 287, 0,
  323, 8, 361, 105, 0,
  230, 148, 202, 312, 0,
  320, 335, 2, 266, 0,
  210, 313, 297, 21, 0,
  269, 82, 115, 0,
  185, 177, 289, 214, 0,
  258, 93, 346, 297, 0,
  175, 37, 312, 0,
  52, 314, 139, 288, 0,
  113, 14, 218, 0,
  113, 132, 114, 168, 0,
  80, 78, 163, 274, 0,
  135, 149, 15, 0
};

/// The circulant values of BG1 for a=5
/// This will be used for all values of Z in the set {5, 10, 20, 40, 80, 160, 320}
/// This is a*2^n with a = 5;
const int16_t k_bg1a5[SimdLdpc::k_maxCirculants] = {
  73, 15, 103, 49, 240, 39, 15, 162, 215, 164, 133, 298, 110, 113, 16, 189, 32, 1, 0,
  303, 294, 27, 261, 161, 133, 4, 80, 129, 300, 76, 266, 72, 83, 260, 301, 0, 0, 0,
  68, 7, 80, 280, 38, 227, 202, 200, 71, 106, 295, 283, 301, 184, 246, 230, 276, 0, 0,
  220, 208, 30, 197, 61, 175, 79, 281, 303, 253, 164, 53, 44, 28, 77, 319, 68, 1, 0,
  233, 205, 0,
  83, 292, 50, 318, 201, 267, 279, 0,
  289, 21, 293, 13, 232, 302, 138, 235, 0,
  12, 88, 207, 50, 25, 76, 0,
  295, 133, 130, 231, 296, 110, 269, 245, 154, 0,
  189, 244, 36, 286, 151, 267, 135, 209, 0,
  14, 80, 211, 75, 161, 311, 0,
  16, 147, 290, 289, 177, 43, 280, 0,
  229, 235, 169, 48, 105, 52, 0,
  39, 302, 303, 160, 37, 0,
  78, 299, 54, 61, 179, 258, 0,
  229, 290, 60, 130, 184, 51, 0,
  69, 140, 45, 115, 300, 0,
  257, 147, 128, 51, 228, 0,
  260, 294, 291, 141, 295, 0,
  64, 181, 101, 270, 41, 0,
  301, 162, 40, 130, 10, 0,
  79, 175, 132, 283, 103, 0,
  177, 20, 55, 316, 0,
  249, 50, 133, 105, 0,
  289, 280, 110, 187, 281, 0,
  172, 295, 96, 46, 0,
  270, 110, 318, 67, 0,
  210, 29, 304, 0,
  11, 293, 50, 234, 0,
  27, 308, 117, 29, 0,
  91, 23, 105, 135, 0,
  222, 308, 66, 162, 0,
  210, 22, 271, 217, 0,
  170, 20, 140, 33, 0,
  187, 296, 5, 44, 0,
  207, 158, 55, 285, 0,
  259, 179, 178, 160, 0,
  298, 15, 115, 0,
  151, 179, 64, 181, 0,
  102, 77, 192, 208, 0,
  32, 80, 197, 0,
  154, 47, 124, 207, 0,
  226, 65, 126, 0,
  228, 69, 176, 102, 0,
  234, 227, 259, 260, 0,
  101, 228, 126, 0
};

/// The circulant values of BG1 for a=7
/// This will be used for all values of Z in the set {7, 14, 28, 56, 112, 224}
/// This is a*2^n with a = 7;
const int16_t k_bg1a7[SimdLdpc::k_maxCirculants] = {
  223, 16, 94, 91, 74, 10, 0, 205, 216, 21, 215, 14, 70, 141, 198, 104, 81, 1, 0,
  141, 45, 151, 46, 119, 157, 133, 87, 206, 93, 79, 9, 118, 194, 31, 187, 0, 0, 0,
  207, 203, 31, 176, 180, 186, 95, 153, 177, 70, 77, 214, 77, 198, 117, 223, 90, 0, 0,
  201, 18, 165, 5, 45, 142, 16, 34, 155, 213, 147, 69, 96, 74, 99, 30, 158, 1, 0,
  170, 10, 0,
  164, 59, 86, 80, 182, 130, 153, 0,
  158, 119, 113, 21, 63, 51, 136, 116, 0,
  17, 76, 104, 100, 150, 158, 0,
  33, 95, 4, 217, 204, 39, 58, 44, 201, 0,
  9, 37, 213, 105, 89, 185, 109, 218, 0,
  82, 165, 174, 19, 194, 103, 0,
  52, 11, 2, 35, 32, 84, 201, 0,
  142, 175, 136, 3, 28, 182, 0,
  81, 56, 72, 217, 78, 0,
  14, 175, 211, 191, 51, 43, 0,
  90, 120, 131, 209, 209, 81, 0,
  154, 164, 43, 189, 101, 0,
  56, 110, 200, 63, 4, 0,
  199, 110, 200, 143, 186, 0,
  8, 6, 103, 198, 8, 0,
  105, 210, 121, 214, 183, 0,
  192, 131, 220, 50, 106, 0,
  53, 0, 3, 148, 0,
  88, 203, 168, 122, 0,
  49, 157, 64, 193, 124, 0,
  1, 166, 65, 81, 0,
  107, 176, 212, 127, 0,
  208, 141, 174, 0,
  146, 153, 217, 114, 0,
  150, 11, 53, 68, 0,
  34, 130, 210, 123, 0,
  175, 49, 177, 128, 0,
  192, 209, 58, 30, 0,
  114, 49, 161, 137, 0,
  82, 186, 68, 150, 0,
  192, 173, 26, 187, 0,
  222, 157, 0, 6, 0,
  81, 195, 138, 0,
  123, 90, 73, 10, 0,
  12, 77, 49, 114, 0,
  67, 45, 96, 0,
  23, 215, 60, 167, 0,
  114, 91, 78, 0,
  206, 22, 134, 161, 0,
  84, 4, 9, 12, 0,
  184, 121, 29, 0
};

/// The circulant values of BG1 for a=9
/// This will be used for all values of Z in the set {9, 18, 36, 72, 144, 288}
/// This is a*2^n with a = 9;
const int16_t k_bg1a9[SimdLdpc::k_maxCirculants] = {
  211, 198, 188, 186, 219, 4, 29, 144, 116, 216, 115, 233, 144, 95, 216, 73, 261, 1, 0,
  179, 162, 223, 256, 160, 76, 202, 117, 109, 15, 72, 152, 158, 147, 156, 119, 0, 0, 0,
  258, 167, 220, 133, 243, 202, 218, 63, 0, 3, 74, 229, 0, 216, 269, 200, 234, 0, 0,
  187, 145, 166, 108, 82, 132, 197, 41, 162, 57, 36, 115, 242, 165, 0, 113, 108, 1, 0,
  246, 235, 0,
  261, 181, 72, 283, 254, 79, 144, 0,
  80, 144, 169, 90, 59, 177, 151, 108, 0,
  169, 189, 154, 184, 104, 164, 0,
  54, 0, 252, 41, 98, 46, 15, 230, 54, 0,
  162, 159, 93, 134, 45, 132, 76, 209, 0,
  178, 1, 28, 267, 234, 201, 0,
  55, 23, 274, 181, 273, 39, 26, 0,
  225, 162, 244, 151, 238, 243, 0,
  231, 0, 216, 47, 36, 0,
  0, 186, 253, 16, 0, 79, 0,
  170, 0, 183, 108, 68, 64, 0,
  270, 13, 99, 54, 0, 0,
  153, 137, 0, 0, 162, 0,
  161, 151, 0, 241, 144, 0,
  0, 0, 118, 144, 0, 0,
  265, 81, 90, 144, 228, 0,
  64, 46, 266, 9, 18, 0,
  72, 189, 72, 257, 0,
  180, 0, 0, 165, 0,
  236, 199, 0, 266, 0, 0,
  205, 0, 0, 183, 0,
  0, 0, 0, 277, 0,
  45, 36, 72, 0,
  275, 0, 155, 62, 0,
  0, 180, 0, 42, 0,
  0, 90, 252, 173, 0,
  144, 144, 166, 19, 0,
  0, 211, 36, 162, 0,
  0, 0, 76, 18, 0,
  197, 0, 108, 0, 0,
  199, 278, 0, 205, 0,
  216, 16, 0, 0, 0,
  72, 144, 0, 0,
  190, 0, 0, 0, 0,
  153, 0, 165, 117, 0,
  216, 144, 2, 0,
  0, 0, 0, 183, 0,
  27, 0, 35, 0,
  52, 243, 0, 270, 0,
  18, 0, 0, 57, 0,
  168, 0, 144, 0
};

/// The circulant values of BG1 for a=11
/// This will be used for all values of Z in the set {11, 22, 44, 88, 176, 352}
/// This is a*2^n with a = 11;
const int16_t k_bg1a11[SimdLdpc::k_maxCirculants] = {
  294, 118, 167, 330, 207, 165, 243, 250, 1, 339, 201, 53, 347, 304, 167, 47, 188, 1, 0,
  77, 225, 96, 338, 268, 112, 302, 50, 167, 253, 334, 242, 257, 133, 9, 302, 0, 0, 0,
  226, 35, 213, 302, 111, 265, 128, 237, 294, 127, 110, 286, 125, 131, 163, 210, 7, 0, 0,
  97, 94, 49, 279, 139, 166, 91, 106, 246, 345, 269, 185, 249, 215, 143, 121, 121, 1, 0,
  42, 256, 0,
  219, 130, 251, 322, 295, 258, 283, 0,
  294, 73, 330, 99, 172, 150, 284, 305, 0,
  3, 103, 224, 297, 215, 39, 0,
  348, 75, 22, 312, 224, 17, 59, 314, 244, 0,
  156, 88, 293, 111, 92, 152, 23, 337, 0,
  175, 253, 27, 231, 49, 267, 0,
  25, 322, 200, 351, 166, 338, 192, 0,
  123, 217, 142, 110, 176, 76, 0,
  311, 251, 265, 94, 81, 0,
  22, 322, 277, 156, 66, 78, 0,
  176, 348, 15, 81, 176, 113, 0,
  190, 293, 332, 331, 114, 0,
  110, 228, 247, 116, 190, 0,
  47, 286, 246, 181, 73, 0,
  87, 110, 147, 258, 204, 0,
  89, 65, 155, 244, 30, 0,
  162, 264, 346, 143, 109, 0,
  280, 157, 236, 113, 0,
  18, 6, 181, 304, 0,
  38, 170, 249, 288, 194, 0,
  279, 255, 111, 54, 0,
  325, 326, 226, 99, 0,
  91, 326, 268, 0,
  102, 1, 40, 167, 0,
  273, 104, 243, 107, 0,
  171, 16, 95, 212, 0,
  101, 297, 279, 222, 0,
  351, 265, 338, 83, 0,
  56, 304, 141, 101, 0,
  60, 320, 112, 54, 0,
  100, 210, 195, 268, 0,
  135, 15, 35, 188, 0,
  319, 236, 85, 0,
  164, 196, 209, 246, 0,
  236, 264, 37, 272, 0,
  304, 237, 135, 0,
  123, 77, 25, 272, 0,
  288, 83, 17, 0,
  210, 3, 53, 167, 0,
  79, 244, 293, 272, 0,
  82, 67, 235, 0
};

/// The circulant values of BG1 for a=13
/// This will be used for all values of Z in the set {13, 26, 52, 104, 208}
/// This is a*2^n with a = 13;
const int16_t k_bg1a13[SimdLdpc::k_maxCirculants] = {
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  22, 11, 124, 0, 10, 0, 0, 2, 16, 60, 0, 6, 30, 0, 168, 31, 105, 0, 0,
  132, 37, 21, 180, 4, 149, 48, 38, 122, 195, 155, 28, 85, 47, 179, 42, 66, 0, 0,
  4, 6, 33, 113, 49, 21, 6, 151, 83, 154, 87, 5, 92, 173, 120, 2, 142, 0, 0,
  24, 204, 0,
  185, 100, 24, 65, 207, 161, 72, 0,
  6, 27, 163, 50, 48, 24, 38, 91, 0,
  145, 88, 112, 153, 159, 76, 0,
  172, 2, 131, 141, 96, 99, 101, 35, 116, 0,
  6, 10, 145, 53, 201, 4, 164, 173, 0,
  126, 77, 156, 16, 12, 70, 0,
  184, 194, 123, 16, 104, 109, 124, 0,
  6, 20, 203, 153, 104, 207, 0,
  52, 147, 1, 16, 46, 0,
  1, 202, 118, 130, 1, 2, 0,
  173, 6, 81, 182, 53, 46, 0,
  88, 198, 160, 122, 182, 0,
  91, 184, 30, 3, 155, 0,
  1, 41, 167, 68, 148, 0,
  12, 6, 166, 184, 191, 0,
  6, 12, 15, 5, 30, 0,
  6, 86, 96, 42, 199, 0,
  44, 58, 130, 131, 0,
  45, 18, 132, 100, 0,
  9, 125, 191, 28, 6, 0,
  4, 74, 16, 28, 0,
  21, 142, 192, 197, 0,
  98, 140, 22, 0,
  4, 1, 40, 93, 0,
  92, 136, 106, 6, 0,
  2, 88, 112, 20, 0,
  4, 49, 125, 194, 0,
  6, 126, 63, 20, 0,
  10, 30, 6, 92, 0,
  4, 153, 197, 155, 0,
  4, 45, 168, 185, 0,
  6, 200, 177, 43, 0,
  82, 2, 135, 0,
  91, 64, 198, 100, 0,
  4, 28, 109, 188, 0,
  10, 84, 12, 0,
  2, 75, 142, 128, 0,
  163, 10, 162, 0,
  1, 163, 99, 98, 0,
  4, 6, 142, 3, 0,
  181, 45, 153, 0
};

/// The circulant values of BG1 for a=15
/// This will be used for all values of Z in the set {15, 30, 60, 120, 240}
/// This is a*2^n with a = 15;
const int16_t k_bg1a15[SimdLdpc::k_maxCirculants] = {
  135, 227, 126, 134, 84, 83, 53, 225, 205, 128, 75, 135, 217, 220, 90, 105, 137, 1, 0,
  96, 236, 136, 221, 128, 92, 172, 56, 11, 189, 95, 85, 153, 87, 163, 216, 0, 0, 0,
  189, 4, 225, 151, 236, 117, 179, 92, 24, 68, 6, 101, 33, 96, 125, 67, 230, 0, 0,
  128, 23, 162, 220, 43, 186, 96, 1, 216, 22, 24, 167, 200, 32, 235, 172, 219, 1, 0,
  64, 211, 0,
  2, 171, 47, 143, 210, 180, 180, 0,
  199, 22, 23, 100, 92, 207, 52, 13, 0,
  77, 146, 209, 32, 166, 18, 0,
  181, 105, 141, 223, 177, 145, 199, 153, 38, 0,
  169, 12, 206, 221, 17, 212, 92, 205, 0,
  116, 151, 70, 230, 115, 84, 0,
  45, 115, 134, 1, 152, 165, 107, 0,
  186, 215, 124, 180, 98, 80, 0,
  220, 185, 154, 178, 150, 0,
  124, 144, 182, 95, 72, 76, 0,
  39, 138, 220, 173, 142, 49, 0,
  78, 152, 84, 5, 205, 0,
  183, 112, 106, 219, 129, 0,
  183, 215, 180, 143, 14, 0,
  179, 108, 159, 138, 196, 0,
  77, 187, 203, 167, 130, 0,
  197, 122, 215, 65, 216, 0,
  25, 47, 126, 178, 0,
  185, 127, 117, 199, 0,
  32, 178, 2, 156, 58, 0,
  27, 141, 11, 181, 0,
  163, 131, 169, 98, 0,
  165, 232, 9, 0,
  32, 43, 200, 205, 0,
  232, 32, 118, 103, 0,
  170, 199, 26, 105, 0,
  73, 149, 175, 108, 0,
  103, 110, 151, 211, 0,
  199, 132, 172, 65, 0,
  161, 237, 142, 180, 0,
  231, 174, 145, 100, 0,
  11, 207, 42, 100, 0,
  59, 204, 161, 0,
  121, 90, 26, 140, 0,
  115, 188, 168, 52, 0,
  4, 103, 30, 0,
  53, 189, 215, 24, 0,
  222, 170, 71, 0,
  22, 127, 49, 125, 0,
  191, 211, 187, 148, 0,
  177, 114, 93, 0
};








/// Basgraph#2 Definitions

/// The positions in the 42 x 52 basegraph matrix of the circulants
/// Each row of text represents a row of the basegraph
/// BG2
const uint8_t k_bg2ColumnPositions[SimdLdpc::k_bg2Circulants] = {
  0, 1, 2, 3, 6, 9, 10, 11,
  0, 3, 4, 5, 6, 7, 8, 9, 11, 12,
  0, 1, 3, 4, 8, 10, 12, 13,
  1, 2, 4, 5, 6, 7, 8, 9, 10, 13,
  0, 1, 11, 14,
  0, 1, 5, 7, 11, 15,
  0, 5, 7, 9, 11, 16,
  1, 5, 7, 11, 13, 17,
  0, 1, 12, 18,
  1, 8, 10, 11, 19,
  0, 1, 6, 7, 20,
  0, 7, 9, 13, 21,
  1, 3, 11, 22,
  0, 1, 8, 13, 23,
  1, 6, 11, 13, 24,
  0, 10, 11, 25,
  1, 9, 11, 12, 26,
  1, 5, 11, 12, 27,
  0, 6, 7, 28,
  0, 1, 10, 29,
  1, 4, 11, 30,
  0, 8, 13, 31,
  1, 2, 32,
  0, 3, 5, 33,
  1, 2, 9, 34,
  0, 5, 35,
  2, 7, 12, 13, 36,
  0, 6, 37,
  1, 2, 5, 38,
  0, 4, 39,
  2, 5, 7, 9, 40,
  1, 13, 41,
  0, 5, 12, 42,
  2, 7, 10, 43,
  0, 12, 13, 44,
  1, 5, 11, 45,
  0, 2, 7, 46,
  10, 13, 47,
  1, 5, 11, 48,
  0, 7, 12, 49,
  2, 10, 13, 50,
  1, 5, 11, 51
};

/// The row-weights of BG2
const uint8_t k_bg2RowWeights[42] = {
  8, 10, 8, 10, 4, 6, 6, 6, 4, 5, 5, 5, 4, 5, 5, 4, 5, 5, 4, 4, 4, 4, 3, 4, 4, 3, 5, 3, 4, 3, 5, 3, 4, 4, 4, 4, 4, 3, 4, 4, 4, 4
};
const int16_t k_bg2RowWeightsCumulative[42] = {
  8, 18, 26, 36, 40, 46, 52, 58, 62, 67, 72, 77, 81, 86, 91, 95, 100, 105, 109, 113, 117, 121, 124, 128, 132, 135, 140, 143, 147, 150, 155, 158, 162, 166, 170, 174, 178, 181, 185, 189, 193, 197
};

/// The circulant values of BG2 for a=2
/// This will be used for all values of Z in the set {2,4,8,16,32,64,128,256}
/// This is a*2^n with a = 2;
const int16_t k_bg2a2[SimdLdpc::k_bg2Circulants] = {
  9, 117, 204, 26, 189, 205, 0, 0,
  167, 166, 253, 125, 226, 156, 224, 252, 0, 0,
  81, 114, 44, 52, 240, 1, 0, 0,
  8, 58, 158, 104, 209, 54, 18, 128, 0, 0,
  179, 214, 71, 0,
  231, 41, 194, 159, 103, 0,
  155, 228, 45, 28, 158, 0,
  129, 147, 140, 3, 116, 0,
  142, 94, 230, 0,
  203, 205, 61, 247, 0,
  11, 185, 0, 117, 0,
  11, 236, 210, 56, 0,
  63, 111, 14, 0,
  83, 2, 38, 222, 0,
  115, 145, 3, 232, 0,
  51, 175, 213, 0,
  203, 142, 8, 242, 0,
  254, 124, 114, 64, 0,
  220, 194, 50, 0,
  87, 20, 185, 0,
  26, 105, 29, 0,
  76, 42, 210, 0,
  222, 63, 0,
  23, 235, 238, 0,
  46, 139, 8, 0,
  228, 156, 0,
  29, 143, 160, 122, 0,
  8, 151, 0,
  98, 101, 135, 0,
  18, 28, 0,
  71, 240, 9, 84, 0,
  106, 1, 0,
  242, 44, 166, 0,
  132, 164, 235, 0,
  147, 85, 36, 0,
  57, 40, 63, 0,
  140, 38, 154, 0,
  219, 151, 0,
  31, 66, 38, 0,
  239, 172, 34, 0,
  0, 75, 120, 0,
  129, 229, 118, 0
};

/// The circulant values of BG2 for a=3
/// This will be used for all values of Z in the set {3,6,12,24,48,96,192,384}
/// This is a*2^n with a = 3;
const int16_t k_bg2a3[SimdLdpc::k_bg2Circulants] = {
  174, 97, 166, 66, 71, 172, 0, 0,
  27, 36, 48, 92, 31, 187, 185, 3, 0, 0,
  25, 114, 117, 110, 114, 1, 0, 0,
  136, 175, 113, 72, 123, 118, 28, 186, 0, 0,
  72, 74, 29, 0,
  10, 44, 121, 80, 48, 0,
  129, 92, 100, 49, 184, 0,
  80, 186, 16, 102, 143, 0,
  118, 70, 152, 0,
  28, 132, 185, 178, 0,
  59, 104, 22, 52, 0,
  32, 92, 174, 154, 0,
  39, 93, 11, 0,
  49, 125, 35, 166, 0,
  19, 118, 21, 163, 0,
  68, 63, 81, 0,
  87, 177, 135, 64, 0,
  158, 23, 9, 6, 0,
  186, 6, 46, 0,
  58, 42, 156, 0,
  76, 61, 153, 0,
  157, 175, 67, 0,
  20, 52, 0,
  106, 86, 95, 0,
  182, 153, 64, 0,
  45, 21, 0,
  67, 137, 55, 85, 0,
  103, 50, 0,
  70, 111, 168, 0,
  110, 17, 0,
  120, 154, 52, 56, 0,
  3, 170, 0,
  84, 8, 17, 0,
  165, 179, 124, 0,
  173, 177, 12, 0,
  77, 184, 18, 0,
  25, 151, 170, 0,
  37, 31, 0,
  84, 151, 190, 0,
  93, 132, 57, 0,
  103, 107, 163, 0,
  147, 7, 60, 0
};

/// The circulant values of BG2 for a=5
/// This will be used for all values of Z in the set {5, 10, 20, 40, 80, 160, 320}
/// This is a*2^n with a = 5;
const int16_t k_bg2a5[SimdLdpc::k_bg2Circulants] = {
  0, 0, 0, 0, 0, 0, 0, 0,
  137, 124, 0, 0, 88, 0, 0, 55, 0, 0,
  20, 94, 99, 9, 108, 1, 0, 0,
  38, 15, 102, 146, 12, 57, 53, 46, 0, 0,
  0, 136, 157, 0,
  0, 131, 142, 141, 64, 0,
  0, 124, 99, 45, 148, 0,
  0, 45, 148, 96, 78, 0,
  0, 65, 87, 0,
  0, 97, 51, 85, 0,
  0, 17, 156, 20, 0,
  0, 7, 4, 2, 0,
  0, 113, 48, 0,
  0, 112, 102, 26, 0,
  0, 138, 57, 27, 0,
  0, 73, 99, 0,
  0, 79, 111, 143, 0,
  0, 24, 109, 18, 0,
  0, 18, 86, 0,
  0, 158, 154, 0,
  0, 148, 104, 0,
  0, 17, 33, 0,
  0, 4, 0,
  0, 75, 158, 0,
  0, 69, 87, 0,
  0, 65, 0,
  0, 100, 13, 7, 0,
  0, 32, 0,
  0, 126, 110, 0,
  0, 154, 0,
  0, 35, 51, 134, 0,
  0, 20, 0,
  0, 20, 122, 0,
  0, 88, 13, 0,
  0, 19, 78, 0,
  0, 157, 6, 0,
  0, 63, 82, 0,
  0, 144, 0,
  0, 93, 19, 0,
  0, 24, 138, 0,
  0, 36, 143, 0,
  0, 2, 55, 0
};

/// The circulant values of BG2 for a=7
/// This will be used for all values of Z in the set {7, 14, 28, 56, 112, 224}
/// This is a*2^n with a = 7;
const int16_t k_bg2a7[SimdLdpc::k_bg2Circulants] = {
  72, 110, 23, 181, 95, 8, 1, 0,
  53, 156, 115, 156, 115, 200, 29, 31, 0, 0,
  152, 131, 46, 191, 91, 0, 0, 0,
  185, 6, 36, 124, 124, 110, 156, 133, 1, 0,
  200, 16, 101, 0,
  185, 138, 170, 219, 193, 0,
  123, 55, 31, 222, 209, 0,
  103, 13, 105, 150, 181, 0,
  147, 43, 152, 0,
  2, 30, 184, 83, 0,
  174, 150, 8, 56, 0,
  99, 138, 110, 99, 0,
  46, 217, 109, 0,
  37, 113, 143, 140, 0,
  36, 95, 40, 116, 0,
  116, 200, 110, 0,
  75, 158, 134, 97, 0,
  48, 132, 206, 2, 0,
  68, 16, 156, 0,
  35, 138, 86, 0,
  6, 20, 141, 0,
  80, 43, 81, 0,
  49, 1, 0,
  156, 54, 134, 0,
  153, 88, 63, 0,
  211, 94, 0,
  90, 6, 221, 6, 0,
  27, 118, 0,
  216, 212, 193, 0,
  108, 61, 0,
  106, 44, 185, 176, 0,
  147, 182, 0,
  108, 21, 110, 0,
  71, 12, 109, 0,
  29, 201, 69, 0,
  91, 165, 55, 0,
  1, 175, 83, 0,
  40, 12, 0,
  37, 97, 46, 0,
  106, 181, 154, 0,
  98, 35, 36, 0,
  120, 101, 81, 0
};

/// The circulant values of BG2 for a=9
/// This will be used for all values of Z in the set {9, 18, 36, 72, 114, 228}
/// This is a*2^n with a = 9;
const int16_t k_bg2a9[SimdLdpc::k_bg2Circulants] = {
  3, 26, 53, 35, 115, 127, 0, 0,
  19, 94, 104, 66, 84, 98, 69, 50, 0, 0,
  95, 106, 92, 110, 111, 1, 0, 0,
  120, 121, 22, 4, 73, 49, 128, 79, 0, 0,
  42, 24, 51, 0,
  40, 140, 84, 137, 71, 0,
  109, 87, 107, 133, 139, 0,
  97, 135, 35, 108, 65, 0,
  70, 69, 88, 0,
  97, 40, 24, 49, 0,
  46, 41, 101, 96, 0,
  28, 30, 116, 64, 0,
  33, 122, 131, 0,
  76, 37, 62, 47, 0,
  143, 51, 130, 97, 0,
  139, 96, 128, 0,
  48, 9, 28, 8, 0,
  120, 43, 65, 42, 0,
  17, 106, 142, 0,
  79, 28, 41, 0,
  2, 103, 78, 0,
  91, 75, 81, 0,
  54, 132, 0,
  68, 115, 56, 0,
  30, 42, 101, 0,
  128, 63, 0,
  142, 28, 100, 133, 0,
  13, 10, 0,
  106, 77, 43, 0,
  133, 25, 0,
  87, 56, 104, 70, 0,
  80, 139, 0,
  32, 89, 71, 0,
  135, 6, 2, 0,
  37, 25, 114, 0,
  60, 137, 93, 0,
  121, 129, 26, 0,
  97, 56, 0,
  1, 70, 1, 0,
  119, 32, 142, 0,
  6, 73, 102, 0,
  48, 47, 19, 0
};

/// The circulant values of BG2 for a=11
/// This will be used for all values of Z in the set {11, 22, 44, 88, 176, 352}
/// This is a*2^n with a = 11;
const int16_t k_bg2a11[SimdLdpc::k_bg2Circulants] = {
  156, 143, 14, 3, 40, 123, 0, 0,
  17, 65, 63, 1, 55, 37, 171, 133, 0, 0,
  98, 168, 107, 82, 142, 1, 0, 0,
  53, 174, 174, 127, 17, 89, 17, 105, 0, 0,
  86, 67, 83, 0,
  79, 84, 35, 103, 60, 0,
  47, 154, 10, 155, 29, 0,
  48, 125, 24, 47, 55, 0,
  53, 31, 161, 0,
  104, 142, 99, 64, 0,
  111, 25, 174, 23, 0,
  91, 175, 24, 141, 0,
  122, 11, 4, 0,
  29, 91, 27, 127, 0,
  11, 145, 8, 166, 0,
  137, 103, 40, 0,
  78, 158, 17, 165, 0,
  134, 23, 62, 163, 0,
  173, 31, 22, 0,
  13, 135, 145, 0,
  128, 52, 173, 0,
  156, 166, 40, 0,
  18, 163, 0,
  110, 132, 150, 0,
  113, 108, 61, 0,
  72, 136, 0,
  36, 38, 53, 145, 0,
  42, 104, 0,
  64, 24, 149, 0,
  139, 161, 0,
  84, 173, 93, 29, 0,
  117, 148, 0,
  116, 73, 142, 0,
  105, 137, 29, 0,
  11, 41, 162, 0,
  126, 152, 172, 0,
  73, 154, 129, 0,
  167, 38, 0,
  112, 7, 19, 0,
  109, 6, 105, 0,
  160, 156, 82, 0,
  132, 6, 8, 0
};

/// The circulant values of BG2 for a=13
/// This will be used for all values of Z in the set {13, 26, 52, 104, 208}
/// This is a*2^n with a = 13;
const int16_t k_bg2a13[SimdLdpc::k_bg2Circulants] = {
  143, 19, 176, 165, 196, 13, 0, 0,
  18, 27, 3, 102, 185, 17, 14, 180, 0, 0,
  126, 163, 47, 183, 132, 1, 0, 0,
  36, 48, 18, 111, 203, 3, 191, 160, 0, 0,
  43, 27, 117, 0,
  136, 49, 36, 132, 62, 0,
  7, 34, 198, 168, 12, 0,
  163, 78, 143, 107, 58, 0,
  101, 177, 22, 0,
  186, 27, 205, 81, 0,
  125, 60, 177, 51, 0,
  39, 29, 35, 8, 0,
  18, 155, 49, 0,
  32, 53, 95, 186, 0,
  91, 20, 52, 109, 0,
  174, 108, 102, 0,
  125, 31, 54, 176, 0,
  57, 201, 142, 35, 0,
  129, 203, 140, 0,
  110, 124, 52, 0,
  196, 35, 114, 0,
  10, 122, 23, 0,
  202, 126, 0,
  52, 170, 13, 0,
  113, 161, 88, 0,
  197, 194, 0,
  164, 172, 49, 161, 0,
  168, 193, 0,
  14, 186, 46, 0,
  50, 27, 0,
  70, 17, 50, 6, 0,
  115, 189, 0,
  110, 0, 163, 0,
  163, 173, 179, 0,
  197, 191, 193, 0,
  157, 167, 181, 0,
  197, 167, 179, 0,
  181, 193, 0,
  157, 173, 191, 0,
  181, 157, 173, 0,
  193, 163, 179, 0,
  191, 197, 167, 0
};

/// The circulant values of BG2 for a=15
/// This will be used for all values of Z in the set {15, 30, 670, 120, 240}
/// This is a*2^n with a = 15;
const int16_t k_bg2a15[SimdLdpc::k_bg2Circulants] = {
  145, 131, 71, 21, 23, 112, 1, 0,
  142, 174, 183, 27, 96, 23, 9, 167, 0, 0,
  74, 31, 3, 53, 155, 0, 0, 0,
  239, 171, 95, 110, 159, 199, 43, 75, 1, 0,
  29, 140, 180, 0,
  121, 41, 169, 88, 207, 0,
  137, 72, 172, 124, 56, 0,
  86, 186, 87, 172, 154, 0,
  176, 169, 225, 0,
  167, 238, 48, 68, 0,
  38, 217, 208, 232, 0,
  178, 214, 168, 51, 0,
  124, 122, 72, 0,
  48, 57, 167, 219, 0,
  82, 232, 204, 162, 0,
  38, 217, 157, 0,
  170, 23, 175, 202, 0,
  196, 173, 195, 218, 0,
  128, 211, 210, 0,
  39, 84, 88, 0,
  117, 227, 6, 0,
  238, 13, 11, 0,
  195, 44, 0,
  5, 94, 111, 0,
  81, 19, 130, 0,
  66, 95, 0,
  146, 66, 190, 86, 0,
  64, 181, 0,
  7, 144, 16, 0,
  25, 57, 0,
  37, 139, 221, 17, 0,
  201, 46, 0,
  179, 14, 116, 0,
  46, 2, 106, 0,
  184, 135, 141, 0,
  85, 225, 175, 0,
  178, 112, 106, 0,
  154, 114, 0,
  42, 41, 105, 0,
  167, 45, 189, 0,
  78, 67, 180, 0,
  53, 215, 230, 0
};