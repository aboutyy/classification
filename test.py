# -*- coding: UTF-8 -*-
__author__ = 'Administrator'
import progressbar
import time

widgets = ['Something: ', progressbar.Percentage(), ' ', progressbar.Bar(),
           ' ', progressbar.ETA(), ' ', progressbar.FileTransferSpeed()]
pbar = progressbar.ProgressBar(widgets=widgets, maxval=10000000).start()
for i in range(1000000):
  pbar.update(10*i+1)
pbar.finish()
