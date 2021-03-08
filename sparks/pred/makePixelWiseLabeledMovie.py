import sys

from pred.utils import PixelWisePredictor

if __name__=='__main__':
    args = sys.argv[1:]
    assert '--model' in args, 'USAGE: --model path_to_model.h5 [--imageid imageid]'

    sparkPredictor = PixelWisePredictor()
    sparkPredictor.generate_prediction_frames(videoSize=1)