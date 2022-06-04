import numpy as np
from energy_fun import energyFunction as ef
class seamFunctions:

    def __init__(self,use_fwd_energy,em_constant,m_threshold):
        self.use_forward_energy=use_fwd_energy
        self.em_constant=em_constant
        self.m_threshold=m_threshold

    def add_seam(self, im, seam_idx):
        
        #Code adapted from https://github.com/vivianhylee/seam-carving.
        
        h, w = im.shape[:2]
        output = np.zeros((h, w + 1, 3))
        for row in range(h):
            col = seam_idx[row]
            for ch in range(3):
                if col == 0:
                    p = np.average(im[row, col: col + 2, ch])
                    output[row, col, ch] = im[row, col, ch]
                    output[row, col + 1, ch] = p
                    output[row, col + 1:, ch] = im[row, col:, ch]
                else:
                    p = np.average(im[row, col - 1: col + 1, ch])
                    output[row, : col, ch] = im[row, : col, ch]
                    output[row, col, ch] = p
                    output[row, col + 1:, ch] = im[row, col:, ch]

        return output


    def add_seam_grayscale(self, im, seam_idx):
        
        # Add a vertical seam to a grayscale image at the indices provided 
        # by averaging the pixels values to the left and right of the seam.
           
        h, w = im.shape[:2]
        output = np.zeros((h, w + 1))
        for row in range(h):
            col = seam_idx[row]
            if col == 0:
                p = np.average(im[row, col: col + 2])
                output[row, col] = im[row, col]
                output[row, col + 1] = p
                output[row, col + 1:] = im[row, col:]
            else:
                p = np.average(im[row, col - 1: col + 1])
                output[row, : col] = im[row, : col]
                output[row, col] = p
                output[row, col + 1:] = im[row, col:]

        return output


    def remove_seam(self, im, boolmask):
        h, w = im.shape[:2]
        boolmask3c = np.stack([boolmask] * 3, axis=2)
        return im[boolmask3c].reshape((h, w - 1, 3))


    def remove_seam_grayscale(self, im, boolmask):
        h, w = im.shape[:2]
        return im[boolmask].reshape((h, w - 1))

    # ========================================================================================================
    # ========================================================================================================
        

    def get_minimum_seam(self, im, mask=None, remove_mask=None):
        
        # DP algorithm for finding the seam of minimum energy. Code adapted from 
        # https://karthikkaranth.me/blog/implementing-seam-carving-with-python/
        
        enFn=ef()
        h, w = im.shape[:2]
        energyfn = enFn.forwardEnergy if self.use_forward_energy else enFn.backwardEnergy
        M = energyfn(im)

        if mask is not None:
            M[np.where(mask > self.m_threshold)] = self.em_constant

        # give removal mask priority over protective mask by using larger negative value
        if remove_mask is not None:
            M[np.where(remove_mask > self.m_threshold)] = -self.em_constant * 100

        backtrack = np.zeros_like(M, dtype=np.int)

        # populate DP matrix
        for i in range(1, h):
            for j in range(0, w):
                if j == 0:
                    idx = np.argmin(M[i - 1, j:j + 2])
                    backtrack[i, j] = idx + j
                    min_energy = M[i-1, idx + j]
                else:
                    idx = np.argmin(M[i - 1, j - 1:j + 2])
                    backtrack[i, j] = idx + j - 1
                    min_energy = M[i - 1, idx + j - 1]

                M[i, j] += min_energy

        # backtrack to find path
        seam_idx = []
        boolmask = np.ones((h, w), dtype=np.bool)
        j = np.argmin(M[-1])
        for i in range(h-1, -1, -1):
            boolmask[i, j] = False
            seam_idx.append(j)
            j = backtrack[i, j]

        seam_idx.reverse()
        return np.array(seam_idx), boolmask
    # ========================================================================================================
    # ========================================================================================================
    def seams_removal(self, im, num_remove, mask=None):
        for _ in range(num_remove):
            seam_idx, boolmask = self.get_minimum_seam(im, mask)
          
            im = self.remove_seam(im, boolmask)
            if mask is not None:
                mask = self.remove_seam_grayscale(mask, boolmask)
        return im, mask

    def seams_insertion(self, im, num_add, mask=None):
        seams_record = []
        temp_im = im.copy()
        temp_mask = mask.copy() if mask is not None else None

        for _ in range(num_add):
            seam_idx, boolmask = self.get_minimum_seam(temp_im, temp_mask)
        

            seams_record.append(seam_idx)
            temp_im = self.remove_seam(temp_im, boolmask)
            if temp_mask is not None:
                temp_mask = self.remove_seam_grayscale(temp_mask, boolmask)

        seams_record.reverse()

        for _ in range(num_add):
            seam = seams_record.pop()
            im = self.add_seam(im, seam)
          
            if mask is not None:
                mask = self.add_seam_grayscale(mask, seam)

            # update the remaining seam indices
            for remaining_seam in seams_record:
                remaining_seam[np.where(remaining_seam >= seam)] += 2         

        return im, mask