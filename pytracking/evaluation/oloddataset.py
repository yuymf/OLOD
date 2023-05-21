import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
from pytracking.utils.load_txt import load_text

def OLODDataset():
    return OLODDatasetClass().get_sequence_list()

class OLODDatasetClass(BaseDataset):

    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.olod_path
        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']

        frames = ['{base_path}/{sequence_path}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path, 
        sequence_path=sequence_path, frame=frame_num, nz=nz, ext=ext) for frame_num in range(start_frame+init_omit, end_frame+1)]

        anno_path = '{}/{}'.format(self.base_path, sequence_info['anno_path'])

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64, backend='numpy')

        return Sequence(sequence_info['name'], frames, ground_truth_rect[init_omit:,:])

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        sequence_info_list = [
            {
                "name": "038",
                "path": "dataseq/038",
                "startFrame": 1,
                "endFrame": 623,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/38_0_622.txt"
            },
            {
                "name": "039",
                "path": "dataseq/039",
                "startFrame": 1,
                "endFrame": 924,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/39_0_923.txt"
            },
            {
                "name": "052",
                "path": "dataseq/052",
                "startFrame": 1,
                "endFrame": 755,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/52_0_754.txt"
            },
            {
                "name": "037",
                "path": "dataseq/037",
                "startFrame": 1,
                "endFrame": 481,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/37_0_480.txt"
            },
            {
                "name": "034",
                "path": "dataseq/034",
                "startFrame": 1,
                "endFrame": 922,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/34_0_921.txt"
            },
            {
                "name": "014",
                "path": "dataseq/014",
                "startFrame": 1,
                "endFrame": 737,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/14_0_736.txt"
            },
            {
                "name": "003",
                "path": "dataseq/003",
                "startFrame": 26,
                "endFrame": 916,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/3_25_915.txt"
            },
            {
                "name": "010",
                "path": "dataseq/010",
                "startFrame": 1,
                "endFrame": 723,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/10_0_722.txt"
            },
            {
                "name": "081",
                "path": "dataseq/081",
                "startFrame": 1,
                "endFrame": 924,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/81_0_923.txt"
            },
            {
                "name": "033",
                "path": "dataseq/033",
                "startFrame": 1,
                "endFrame": 1044,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/33_0_1043.txt"
            },
            {
                "name": "083",
                "path": "dataseq/083",
                "startFrame": 1,
                "endFrame": 760,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/83_0_759.txt"
            },
            {
                "name": "007",
                "path": "dataseq/007",
                "startFrame": 1,
                "endFrame": 760,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/7_0_759.txt"
            },
            {
                "name": "035",
                "path": "dataseq/035",
                "startFrame": 1,
                "endFrame": 1207,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/35_0_1206.txt"
            },
            {
                "name": "002",
                "path": "dataseq/002",
                "startFrame": 1,
                "endFrame": 904,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/2_0_903.txt"
            },
            {
                "name": "072",
                "path": "dataseq/072",
                "startFrame": 1,
                "endFrame": 676,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/72_0_675.txt"
            },
            {
                "name": "075",
                "path": "dataseq/075",
                "startFrame": 1,
                "endFrame": 717,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/75_0_716.txt"
            },
            {
                "name": "057",
                "path": "dataseq/057",
                "startFrame": 1,
                "endFrame": 1353,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/57_0_1352.txt"
            },
            {
                "name": "078",
                "path": "dataseq/078",
                "startFrame": 1,
                "endFrame": 579,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/78_0_578.txt"
            },
            {
                "name": "005",
                "path": "dataseq/005",
                "startFrame": 1,
                "endFrame": 918,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/5_0_917.txt"
            },
            {
                "name": "046",
                "path": "dataseq/046",
                "startFrame": 33,
                "endFrame": 737,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/46_32_736.txt"
            },
            {
                "name": "070",
                "path": "dataseq/070",
                "startFrame": 243,
                "endFrame": 1408,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/70_242_1407.txt"
            },
            {
                "name": "058",
                "path": "dataseq/058",
                "startFrame": 1,
                "endFrame": 627,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/58_0_626.txt"
            },
            {
                "name": "024_1",
                "path": "dataseq/024",
                "startFrame": 1,
                "endFrame": 662,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/24_0_661.txt"
            },
            {
                "name": "036",
                "path": "dataseq/036",
                "startFrame": 1,
                "endFrame": 414,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/36_0_413.txt"
            },
            {
                "name": "060",
                "path": "dataseq/060",
                "startFrame": 1,
                "endFrame": 896,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/60_0_895.txt"
            },
            {
                "name": "021",
                "path": "dataseq/021",
                "startFrame": 1,
                "endFrame": 951,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/21_0_950.txt"
            },
            {
                "name": "044",
                "path": "dataseq/044",
                "startFrame": 1,
                "endFrame": 920,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/44_0_919.txt"
            },
            {
                "name": "001",
                "path": "dataseq/001",
                "startFrame": 1,
                "endFrame": 690,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/1_0_689.txt"
            },
            {
                "name": "024_2",
                "path": "dataseq/024",
                "startFrame": 1,
                "endFrame": 688,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/24_0_687.txt"
            },
            {
                "name": "076",
                "path": "dataseq/076",
                "startFrame": 61,
                "endFrame": 1413,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/76_60_1412.txt"
            },
            {
                "name": "006",
                "path": "dataseq/006",
                "startFrame": 1,
                "endFrame": 993,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/6_0_992.txt"
            },
            {
                "name": "023_1",
                "path": "dataseq/023",
                "startFrame": 1,
                "endFrame": 326,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/23_0_325.txt"
            },
            {
                "name": "063",
                "path": "dataseq/063",
                "startFrame": 1,
                "endFrame": 935,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/63_0_934.txt"
            },
            {
                "name": "028",
                "path": "dataseq/028",
                "startFrame": 1,
                "endFrame": 727,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/28_0_726.txt"
            },
            {
                "name": "062",
                "path": "dataseq/062",
                "startFrame": 1,
                "endFrame": 619,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/62_0_618.txt"
            },
            {
                "name": "049",
                "path": "dataseq/049",
                "startFrame": 1,
                "endFrame": 805,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/49_0_804.txt"
            },
            {
                "name": "017",
                "path": "dataseq/017",
                "startFrame": 1,
                "endFrame": 787,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/17_0_786.txt"
            },
            {
                "name": "071",
                "path": "dataseq/071",
                "startFrame": 1,
                "endFrame": 715,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/71_0_714.txt"
            },
            {
                "name": "080",
                "path": "dataseq/080",
                "startFrame": 1,
                "endFrame": 709,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/80_0_708.txt"
            },
            {
                "name": "012",
                "path": "dataseq/012",
                "startFrame": 1,
                "endFrame": 872,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/12_0_871.txt"
            },
            {
                "name": "004",
                "path": "dataseq/004",
                "startFrame": 1,
                "endFrame": 966,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/4_0_965.txt"
            },
            {
                "name": "053",
                "path": "dataseq/053",
                "startFrame": 1,
                "endFrame": 819,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/53_0_818.txt"
            },
            {
                "name": "061",
                "path": "dataseq/061",
                "startFrame": 1,
                "endFrame": 940,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/61_0_939.txt"
            },
            {
                "name": "011",
                "path": "dataseq/011",
                "startFrame": 1,
                "endFrame": 879,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/11_0_878.txt"
            },
            {
                "name": "068",
                "path": "dataseq/068",
                "startFrame": 307,
                "endFrame": 826,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/68_306_825.txt"
            },
            {
                "name": "069",
                "path": "dataseq/069",
                "startFrame": 1,
                "endFrame": 473,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/69_0_472.txt"
            },
            {
                "name": "025",
                "path": "dataseq/025",
                "startFrame": 1,
                "endFrame": 816,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/25_0_815.txt"
            },
            {
                "name": "031",
                "path": "dataseq/031",
                "startFrame": 1,
                "endFrame": 363,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/31_0_362.txt"
            },
            {
                "name": "022",
                "path": "dataseq/022",
                "startFrame": 1,
                "endFrame": 909,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/22_0_908.txt"
            },
            {
                "name": "026",
                "path": "dataseq/026",
                "startFrame": 1,
                "endFrame": 868,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/26_0_867.txt"
            },
            {
                "name": "073",
                "path": "dataseq/073",
                "startFrame": 1,
                "endFrame": 425,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/73_0_424.txt"
            },
            {
                "name": "048",
                "path": "dataseq/048",
                "startFrame": 1,
                "endFrame": 319,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/48_0_318.txt"
            },
            {
                "name": "074",
                "path": "dataseq/074",
                "startFrame": 1,
                "endFrame": 695,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/74_0_694.txt"
            },
            {
                "name": "015",
                "path": "dataseq/015",
                "startFrame": 1,
                "endFrame": 658,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/15_0_657.txt"
            },
            {
                "name": "082",
                "path": "dataseq/082",
                "startFrame": 1,
                "endFrame": 884,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/82_0_883.txt"
            },
            {
                "name": "008",
                "path": "dataseq/008",
                "startFrame": 1,
                "endFrame": 955,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/8_0_954.txt"
            },
            {
                "name": "032",
                "path": "dataseq/032",
                "startFrame": 1,
                "endFrame": 994,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/32_0_993.txt"
            },
            {
                "name": "045",
                "path": "dataseq/045",
                "startFrame": 107,
                "endFrame": 543,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/45_106_542.txt"
            },
            {
                "name": "016",
                "path": "dataseq/016",
                "startFrame": 1,
                "endFrame": 481,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/16_0_480.txt"
            },
            {
                "name": "023_2",
                "path": "dataseq/023",
                "startFrame": 1,
                "endFrame": 487,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/23_0_486.txt"
            },
            {
                "name": "013",
                "path": "dataseq/013",
                "startFrame": 1,
                "endFrame": 981,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/13_0_980.txt"
            },
            {
                "name": "077",
                "path": "dataseq/077",
                "startFrame": 1,
                "endFrame": 796,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/77_0_795.txt"
            },
            {
                "name": "027",
                "path": "dataseq/027",
                "startFrame": 1,
                "endFrame": 991,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/27_0_990.txt"
            },
            {
                "name": "079",
                "path": "dataseq/079",
                "startFrame": 1,
                "endFrame": 801,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/79_0_800.txt"
            },
            {
                "name": "029",
                "path": "dataseq/029",
                "startFrame": 1,
                "endFrame": 1214,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/29_0_1213.txt"
            },
            {
                "name": "009",
                "path": "dataseq/009",
                "startFrame": 1,
                "endFrame": 796,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/9_0_795.txt"
            },
            {
                "name": "043",
                "path": "dataseq/043",
                "startFrame": 1,
                "endFrame": 1353,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/43_0_1352.txt"
            },
            {
                "name": "066",
                "path": "dataseq/066",
                "startFrame": 33,
                "endFrame": 707,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/66_32_706.txt"
            },
            {
                "name": "047",
                "path": "dataseq/047",
                "startFrame": 1,
                "endFrame": 775,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/47_0_774.txt"
            },
            {
                "name": "030",
                "path": "dataseq/030",
                "startFrame": 1,
                "endFrame": 744,
                "nz": 6,
                "ext": "JPG",
                "anno_path": "anno/30_0_743.txt"
            }
        ]
        
        return sequence_info_list