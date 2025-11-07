import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt

# Korrekt filväg
# tiff_path = "C:/Sample_Maxar/200006031532_01/200006031532_01_P001_PSH/24SEP22234650-S2AS-200006031532_01_P001.TIF"
# tiff_path = "C:/Users/igunnard/HD15_WO_000043025_1_1_SAL22036285-1_ACQ_PNEO3_00816400750989.TIF"
tiff_path = "WVP.TIF"

# Läs och visa bilden
with rasterio.open(tiff_path) as src:
    print("Metadata:", src.meta)
    show(src, title="Förhandsgranskning av GeoTIFF")
    plt.show()

# with rasterio.open(tiff_path) as src:
#     r = src.read(4)  # Röd kanal
#     g = src.read(3)  # Grön kanal
#     b = src.read(2)  # Blå kanal

#     rgb = np.stack((r, g, b), axis=-1)
#     plt.imshow(rgb / 10000)  # Normalisera om värden är höga
#     plt.title("RGB-komposit")
#     plt.axis("off")
    # plt.show()
    
    
# from rio_cogeo.cogeo import cog_translate
# from rio_cogeo.profiles import cog_profiles
# from rasterio import open as rio_open
# from rasterio.enums import Resampling

# # Ange sökvägar
# input_path = "C:/Sample_Maxar/200006031532_01/200006031532_01_P001_PSH/24SEP22234650-S2AS-200006031532_01_P001.TIF"
# output_path = "output_cog.tif"

# # Välj en COG-profil (t.ex. för överföring via nätverk)
# profile = cog_profiles.get("deflate")

# # Öppna originalfilen
# with rio_open(input_path) as src_dst:
#     cog_translate(
#         src_dst,
#         output_path,
#         profile,
#         in_memory=True,  # snabbar upp processen
#         config={
#             "GDAL_TIFF_INTERNAL_MASK": True,
#             "NUM_THREADS": "ALL_CPUS",
#             "GDAL_TIFF_OVR_BLOCKSIZE": "128"
#         },
#         quiet=False
#     )
