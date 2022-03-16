# VolcanicAshInversion
This repository contains the volcanic ash inversion source code written in Python. 

The source code is used to take a set of forward runs with unit emissions, and then find the combination of these that matches observations from sattelite images best. 

The code first generates an a priori estimate of ash emissions from observations of ash cloud heihgt. It then continues by colocating ash simulatinons with the observations, and finally assemble these source (forward simulation) - receptor (satellite image) relationships into a large source-receptor matrix. We then use least squares with Tikhonov regularization to find the emissions that match the observatios best. 
