import pandas as pd
import matplotlib.pyplot as plt
import os

from deepcardio_utils import ImageReader, get_mask, plot_cell

if __name__=='__main__':
    imageReader = ImageReader()
    images = imageReader.get_full_images()
    confsDF, detSparksDF = imageReader.get_spark_simple_data()

    plotsFolderPath = os.path.join(imageReader.get_image_folder(), imageReader.get_image_id() + '_xyt_size')
    if not os.path.exists(plotsFolderPath):
        os.makedirs(plotsFolderPath)

    xytList = []
    for sparkIdx in range(int(confsDF.loc[0, 'Surviving sparks'])):
        frameIni = int(detSparksDF.loc[sparkIdx, :].tolist()[3]) - 25
        frameFin = int(detSparksDF.loc[sparkIdx, :].tolist()[3]) + 25
        sparkX = int(detSparksDF.loc[sparkIdx, 'Xpix'])
        sparkY = int(detSparksDF.loc[sparkIdx, 'Ypix'])
        maskSize = detSparksDF.loc[sparkIdx, 'FWHM'] / float(confsDF.loc[:, 'Pixel size(um)'])
        mask = get_mask(images.shape[1], images.shape[2], sparkX, sparkY, maskSize)

        res = []
        for i in range(frameIni, frameFin + 1):
            res.append(images[i][:, :, 2][mask].mean())

        avgS = pd.Series(res)
        rollS = avgS.rolling(3, center=True).mean()
        sparkFrameIdx = rollS.idxmax() + frameIni

        plt.plot(avgS)
        plt.plot(rollS)
        plt.scatter(rollS.idxmax(), rollS.max(), c='red')
        plt.title(f"Spark idx {sparkIdx} on frame {sparkFrameIdx}")
        plt.savefig(os.path.join(plotsFolderPath, f"{sparkIdx}.png"))
        plt.close()

        tIni = max(sparkFrameIdx-1, frameIni)
        tFin = min(sparkFrameIdx+1, frameFin)
        xytList.append((sparkX, sparkY, tIni, tFin, maskSize))

    xytDF = pd.DataFrame(xytList, columns=['x', 'y', 'tIni', 'tFin', 'pixelSize'])
    storePath = plotsFolderPath + '.csv'
    xytDF.to_csv(storePath, index=False)