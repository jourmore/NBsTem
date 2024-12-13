# Copyright 2024 XueMei-Pu Lab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__copyright__    = 'Copyright (C) 2024 XuMei-Pu Lab, SiChuan University'
__version__      = '1.0'
__license__      = 'Apache License 2.0'
__author__       = 'Jourmore'
__author_email__ = 'maojun@stu.scu.edu.cn'
__url__          = 'http://www.nbscal.online/'
__about__        = 'NBsTem: A webserver for Nanobody Thermostability Prediction.'

from .ResLSTM_C import Network as nbstem_ResLSTM
from .CNN_R import Network as nbstem_CNN
