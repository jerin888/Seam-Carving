
import numpy as np
from seam_fun import seamFunctions as sf


class SeamCarver:
    def __init__(self,im,use_forward_energy,em_constant=1000000.0,m_threshold=10):
        self.image=im
        self.m_threshold=m_threshold
        self.seamObj=sf(use_forward_energy,em_constant,m_threshold)

        

    def rotateImage(self,image, clockwise):
        k = 1 if clockwise else 3
        return np.rot90(image, k)   
    

    #Seam Carve Driver function
    def seamCarve(self, im, dy, dx, mask=None):

        im = im.astype(np.float64)
        h, w = im.shape[:2]
        print(h + dy, w + dx, dy,h,dx,w)
        assert h + dy > 0 and w + dx > 0 and dy <= h and dx <= w

        if mask is not None:
            mask = mask.astype(np.float64)

        output = im

        if dx < 0:
            output, mask = self.seamObj.seams_removal(output, -dx, mask)

        elif dx > 0:
            output, mask = self.seamObj.seams_insertion(output, dx, mask)

        if dy < 0:
            output = self.rotateImage(output, True)
            if mask is not None:
                mask = self.rotateImage(mask, True)
            output, mask = self.seamObj.seams_removal(output, -dy, mask)
            output = self.rotateImage(output, False)

        elif dy > 0:
            output = self.rotateImage(output, True)
            if mask is not None:
                mask = self.rotateImage(mask, True)
            output, mask = self.seamObj.seams_insertion(output, dy, mask)
            output = self.rotateImage(output, False)

        return output

    #Object Removal Driver Function
    def objectRemoval(self, im, rmask, mask=None):
        im = im.astype(np.float64)
        rmask = rmask.astype(np.float64)
        if mask is not None:
            mask = mask.astype(np.float64)
        output = im

        h, w = im.shape[:2]

        while len(np.where(rmask > self.m_threshold)[0]) > 0:
            seam_idx, boolmask = self.seamObj.get_minimum_seam(output, mask, rmask)
                     
            output = self.seamObj.remove_seam(output, boolmask)
            rmask = self.seamObj.remove_seam_grayscale(rmask, boolmask)
            if mask is not None:
                mask = self.seamObj.remove_seam_grayscale(mask, boolmask)

        num_add = w - output.shape[1]
        output, mask = self.seamObj.seams_insertion(output, num_add, mask)

        return output 