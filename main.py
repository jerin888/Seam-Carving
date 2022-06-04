"""
Command Examples:


Object Resize using Backward Energy without Input Mask:
 python3 main.py --resize --input in/landscape.jpg --output out/landscape_op_be --nh 300 --nw 400

Image Resize using Backward Energy with Input Mask:
 python3 main.py --resize --input in/car.jpg --output out/car_op_mask_fe.jpg  --nh 500 --nw 600 --forward_energy --input_mask in/car_mask.jpg
 
Image Resize using Forward Energy:
 python3 main.py --remove --input ./in/Bros.jpg --output ./out/bros_i_fe_remove.jpg --input_mask ./in/Bros_i_mask.jpg --remove_mask ./in/Bros_r_mask.jpg --forward_energy

Object Removal Using Backward Energy and Remove Mask:
 python3 main.py --remove --input ./in/birds.jpeg --output ./out/birds_be_remove.jpeg --remove_mask ./in/birds_mask.jpg

Object Removal Using Forward Energy and Remove Mask:
 python3 main.py --remove --input ./in/fed.jpeg --output ./out/fed_fe_remove.jpeg --remove_mask ./in/fed_mask.jpg --forward_energy

"""

import cv2
import argparse

from seamCarver import SeamCarver



SHOULD_DOWNSIZE = True                    # if True, downsize image for faster carving
DOWNSIZE_WIDTH = 500                      # resized image width if SHOULD_DOWNSIZE is True



# resize image for faster processing. Aspect ratio of 4:3 is maintained with new width set to 500.
def resize(image, width):
    dim = None
    dim = (width, int(3 * width / 4))
    return cv2.resize(image, dim)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Implements seam carving to resize image using forward and backward energy. Also performs object detection and removal using masks ')
    parser.add_argument("--resize", action='store_true')
    parser.add_argument("--remove", action='store_true')

    parser.add_argument("--input", help="Image Path", required=True)
    parser.add_argument("--output", help="Output file name", required=True)
    parser.add_argument("--input_mask", default='',help="Path to (protective) mask")
    parser.add_argument("--remove_mask", help="Path to removal mask")
    parser.add_argument("--nh", help="New Hieght", type=int, default=0)
    parser.add_argument("--nw", help="New Width", type=int, default=0)
 
    parser.add_argument("--forward_energy", help="Use forward energy map", action='store_true')
    args = parser.parse_args()



    im = cv2.imread(args.input)
    assert im is not None
    mask = cv2.imread(args.input_mask, 0) if args.input_mask else None
    rmask = cv2.imread(args.remove_mask, 0) if args.remove_mask else None

    

    # downsize image for faster processing
    h, w = im.shape[:2]
    if SHOULD_DOWNSIZE and w > DOWNSIZE_WIDTH:
        im = resize(im, width=DOWNSIZE_WIDTH)
        if mask is not None:
            mask = resize(mask, width=DOWNSIZE_WIDTH)
        if rmask is not None:
            rmask = resize(rmask, width=DOWNSIZE_WIDTH)
    h, w = im.shape[:2]

    op = SeamCarver(im,args.forward_energy)
    # image resize mode
   
    if args.resize:

        dr, dc = int(args.nh - h), int(args.nw - w)
        assert dr is not None and dc is not None
        
        output = op.seamCarve(im, dr, dc, mask)
        cv2.imwrite(args.output, output)


    
    # object removal mode
    elif args.remove:
        assert rmask is not None
        output = op.objectRemoval(im, rmask, mask)
        cv2.imwrite(args.output, output)