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

#include "divide.h"

#include <gtest/gtest.h>

TEST(DivideCheck, TestCeili)
{
    ASSERT_EQ(ceili(28, 32), 1);

    ASSERT_EQ(ceili(32, 32), 1);

    ASSERT_EQ(ceili(33, 32), 2);
}

TEST(DivideCheck, TestFloori)
{
    ASSERT_EQ(floori(28, 32), 0);

    ASSERT_EQ(floori(32, 32), 1);

    ASSERT_EQ(floori(33, 32), 1);
}
