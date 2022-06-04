import numpy as np
from scipy import ndimage as ndi
import cv2
class energyFunction:

    def backwardEnergy(self,im):
          
            #Simple gradient magnitude energy map.
            
            xgrad = ndi.convolve1d(im, np.array([1, 0, -1]), axis=1, mode='wrap')
            ygrad = ndi.convolve1d(im, np.array([1, 0, -1]), axis=0, mode='wrap')
            
            grad_mag = np.sqrt(np.sum(xgrad**2, axis=2) + np.sum(ygrad**2, axis=2))

            return grad_mag


    def forwardEnergy(self,im):
            
            # Forward energy algorithm from "Improved Seam Carving for Video Retargeting"
            # by Rubinstein, Shamir, Avidan.
            # Vectorized code adapted from
            # https://github.com/axu2/improved-seam-carving.
            
            h, w = im.shape[:2]
            im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)

            energy = np.zeros((h, w))
            m = np.zeros((h, w))
            
            U = np.roll(im, 1, axis=0)
            L = np.roll(im, 1, axis=1)
            R = np.roll(im, -1, axis=1)
            
            cU = np.abs(R - L)
            cL = np.abs(U - L) + cU
            cR = np.abs(U - R) + cU
            
            for i in range(1, h):
                mU = m[i-1]
                mL = np.roll(mU, 1)
                mR = np.roll(mU, -1)
                
                mULR = np.array([mU, mL, mR])
                cULR = np.array([cU[i], cL[i], cR[i]])
                mULR += cULR

                argmins = np.argmin(mULR, axis=0)
                m[i] = np.choose(argmins, mULR)
                energy[i] = np.choose(argmins, cULR)
            
                    
                
            return energy