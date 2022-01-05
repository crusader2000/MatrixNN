#!/usr/bin/env python
# coding: utf-8

from util.conf_util import *

import sys
import pickle
import glob
import os
import logging
import time
from datetime import datetime
from datetime import date
import random


if __name__ == "__main__":
	if len(sys.argv) == 2:
		conf_name = sys.argv[1]
		print("train conf_name:", conf_name)
		conf = get_default_conf(f"./config/{conf_name}.yaml")
	else:
		print("default")
		conf = get_default_conf()


