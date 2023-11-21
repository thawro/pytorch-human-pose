"""Download all the data needed in the project"""
from geda.data_providers.mpii import MPIIDataProvider
from geda.data_providers.coco import COCOKeypointsDataProvider
from src.utils.config import DS_ROOT
from src.logging.pylogger import get_pylogger

log = get_pylogger(__name__)


def main():
    log.info("Preparing MPII dataset")
    mpii_dataprovider = MPIIDataProvider(root=str(DS_ROOT / "MPII"))
    mpii_dataprovider.get_data(remove_zip=True)

    log.info("Preparing COCO dataset")
    coco_dataprovider = COCOKeypointsDataProvider(root=str(DS_ROOT / "COCO"))
    coco_dataprovider.get_data(remove_zip=True)


if __name__ == "__main__":
    main()
