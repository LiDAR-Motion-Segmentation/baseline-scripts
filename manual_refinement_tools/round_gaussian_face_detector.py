import hydra
from omegaconf import DictConfig
import os
import cv2
import logging
from pathlib import Path
from tqdm import tqdm

from blur_engine import BlurEngine
from face_detector import FaceDetector

log = logging.getLogger(__name__)


class ProcessingPipeline:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.detector = FaceDetector(
            model_path=cfg.model.path,
            device=cfg.system.device,
            conf_thresh=cfg.model.conf_threshold,
        )
        self.blur_engine = BlurEngine(
            intensity=cfg.blur.intensity, feather_ratio=cfg.blur.feather_ratio
        )
        self.input_dir = Path(cfg.io.input_dir)
        self.output_dir = Path(cfg.io.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        images = list(
            self.input_dir.glob("*.[jJ][pP]*[gG]")
        )  # Matches .jpg, .jpeg, .png variants
        if not images:
            log.warning(f"No images found in {self.input_dir}")
            return

        log.info(
            f"Starting processing of {len(images)} images on {self.cfg.system.device}"
        )

        # TQDM provides a progress bar
        for img_path in tqdm(images, desc="Anonymizing"):
            try:
                self.process_single_image(img_path)
            except Exception as e:
                log.error(f"Error processing {img_path.name}: {e}")

    def process_single_image(self, img_path: Path):
        img = cv2.imread(str(img_path))
        if img is None:
            return

        faces = self.detector.detect(str(img_path))

        # Apply Blur to detected faces
        for x, y, w, h in faces:
            img = self.blur_engine.apply_round_blur(img, x, y, w, h)

        save_path = self.output_dir / img_path.name
        cv2.imwrite(str(save_path), img)


@hydra.main(
    version_base=None, config_path="../config", config_name="blur_people_face_config"
)
def main(cfg: DictConfig):
    pipeline = ProcessingPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
