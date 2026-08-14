import os
import shutil
from glob import glob

# Thu muc chua checkpoint xuat ra de dung ngoai (deploy / pretrain).
# Tach rieng vi day KHONG phai state day du cua Fabric -> khong resume duoc.
EXPORT_DIRNAME = "export"


class Top_K_results_manager:
    """
    hold a results_path list to ke

    """

    def __init__(self, mode="max", max_len=5):
        self.max_len = max_len
        self.mode = mode
        self.results_path = []

    def update(self, output_path, metric):
        self.results_path.append((output_path, metric))
        if len(self.results_path) > self.max_len:
            self._sort_and_remove()

    def _sort_and_remove(self):
        self.results_path.sort(key=lambda x: x[1], reverse=self.mode == "max")
        try:
            shutil.rmtree(self.results_path.pop()[0])
        except Exception as e:
            print(f"Error: {e}")


def auto_resume_helper(output_dir, logger):
    checkpoints = glob(os.path.join(output_dir, "**", "*.pth"), recursive=True)
    # Bo qua export/: deploy va pretrain ckpt khong chua optimizer/scheduler,
    # resume tu chung se im lang mat trang thai training.
    checkpoints = [
        c for c in checkpoints if EXPORT_DIRNAME not in os.path.normpath(c).split(os.sep)
    ]
    logger.info(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max(checkpoints, key=os.path.getmtime)
        logger.info(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file
