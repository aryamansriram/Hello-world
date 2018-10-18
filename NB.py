# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 15:17:49 2018

@author: Aryaman Sriram
"""

import numpy as np
import os
os.getcwd()
cd("C:\Users\Aryaman Sriram\Documents\email")
def textparse(strin):
    import re
    LOS=re.split(r'\W*',strin)
    return [tok.lower() for tok in LOS if len(tok)>2]
            