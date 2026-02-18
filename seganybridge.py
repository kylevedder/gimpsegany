"""
Script to generate Meta Segment Anything masks.

Adapted from:
https://github.com/facebookresearch/segment-anything-2
https://github.com/facebookresearch/segment-anything

Author: Shrinivas Kulkarni

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""

import torch
import numpy as np
import cv2
import sys
import os
import time
import logging

# SAM2 imports
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# SAM1 imports
from segment_anything import (
    sam_model_registry,
    SamAutomaticMaskGenerator as SamAutomaticMaskGenerator_SAM1,
    SamPredictor,
)

SAM3_DEFAULT_IMGSZ = 1036
MAX_INPUT_DIM = 2048

_orig_image_size = None  # (h, w) set by main() if image was downscaled

# SAM3 imports (via ultralytics, which includes MPS support)
try:
    from ultralytics import SAM as UltralyticsSAM
    from ultralytics.models.sam import SAM3SemanticPredictor
    HAS_SAM3 = True
except ImportError:
    HAS_SAM3 = False

# --- Utility Functions ---


def packBoolArray(filepath, arr):
    packed_data = bytearray()
    num_rows = len(arr)
    num_cols = len(arr[0])
    packed_data.extend(
        [num_rows >> 24, (num_rows >> 16) & 255, (num_rows >> 8) & 255, num_rows & 255]
    )
    packed_data.extend(
        [num_cols >> 24, (num_cols >> 16) & 255, (num_cols >> 8) & 255, num_cols & 255]
    )
    current_byte = 0
    bit_position = 0
    for row in arr:
        for boolean_value in row:
            if boolean_value:
                current_byte |= 1 << bit_position
            bit_position += 1
            if bit_position == 8:
                packed_data.append(current_byte)
                current_byte = 0
                bit_position = 0
    if bit_position > 0:
        packed_data.append(current_byte)
    with open(filepath, "wb") as f:
        f.write(packed_data)
    return packed_data


def saveMask(filepath, maskArr, formatBinary):
    if formatBinary:
        packBoolArray(filepath, maskArr)
    else:
        with open(filepath, "w") as f:
            for row in maskArr:
                f.write("".join(str(int(val)) for val in row) + "\n")


def saveMasks(masks, saveFileNoExt, formatBinary):
    if len(masks) == 0:
        logging.warning("No masks to save")
        return
    for i, mask in enumerate(masks):
        mask = np.array(mask)
        if _orig_image_size is not None:
            orig_h, orig_w = _orig_image_size
            mask = cv2.resize(
                mask.astype(np.uint8),
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            )
        filepath = saveFileNoExt + str(i) + ".seg"
        h, w = mask.shape[:2]
        logging.info(f"Saving mask {i}: {w}x{h} -> {filepath}")
        arr = [[val for val in row] for row in mask]
        saveMask(filepath, arr, formatBinary)
    logging.info(f"Saved {len(masks)} mask(s) total")


# --- Strategy Pattern Implementation ---


class SegmentationStrategy:
    def get_model_type_from_filename(self, model_filename):
        raise NotImplementedError

    def load_model(self, checkPtFilePath, modelType):
        raise NotImplementedError

    def segment_auto(self, sam, cvImage, saveFileNoExt, formatBinary, **kwargs):
        raise NotImplementedError

    def segment_box(self, sam, cvImage, maskType, boxCos, saveFileNoExt, formatBinary, imgsz=SAM3_DEFAULT_IMGSZ):
        raise NotImplementedError

    def segment_sel(
        self, sam, cvImage, maskType, selFile, boxCos, saveFileNoExt, formatBinary, imgsz=SAM3_DEFAULT_IMGSZ
    ):
        raise NotImplementedError

    def segment_text(self, sam, cvImage, saveFileNoExt, formatBinary, textPrompt, imgsz=SAM3_DEFAULT_IMGSZ):
        raise NotImplementedError

    def run_test(self, sam):
        raise NotImplementedError

    def cleanup(self):
        pass


class SAM1Strategy(SegmentationStrategy):
    MODEL_TYPE_LOOKUP = {
        "sam_vit_h_4b8939": "vit_h",
        "sam_vit_l_0b3195": "vit_l",
        "sam_vit_b_01ec64": "vit_b",
    }

    def get_model_type_from_filename(self, model_filename):
        filename_stem = os.path.splitext(model_filename)[0]
        model_type = self.MODEL_TYPE_LOOKUP.get(filename_stem)
        if model_type:
            print(f"Auto-detected SAM1 model type: {model_type}")
            return model_type
        else:
            print(
                f"Error: Could not auto-detect model type from SAM1 filename: {model_filename}"
            )
            print(
                f"Please use one of the following file names: {list(self.MODEL_TYPE_LOOKUP.keys())}"
            )
            return None

    def load_model(self, checkPtFilePath, modelType):
        try:
            sam = sam_model_registry[modelType](checkpoint=checkPtFilePath)
            print("SAM1 Model loaded successfully!")
            return sam
        except Exception as e:
            print(f"Error loading SAM1 model: {e}")
            return None

    def segment_auto(self, sam, cvImage, saveFileNoExt, formatBinary, **kwargs):
        mask_generator = SamAutomaticMaskGenerator_SAM1(sam)
        masks = mask_generator.generate(cvImage)
        masks = [mask["segmentation"] for mask in masks]
        saveMasks(masks, saveFileNoExt, formatBinary)

    def segment_box(self, sam, cvImage, maskType, boxCos, saveFileNoExt, formatBinary, imgsz=SAM3_DEFAULT_IMGSZ):
        predictor = SamPredictor(sam)
        predictor.set_image(cvImage)
        input_box = np.array(boxCos)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box,
            multimask_output=(maskType == "Multiple"),
        )
        saveMasks(masks, saveFileNoExt, formatBinary)

    def segment_sel(
        self, sam, cvImage, maskType, selFile, boxCos, saveFileNoExt, formatBinary, imgsz=SAM3_DEFAULT_IMGSZ
    ):
        pts = []
        with open(selFile, "r") as f:
            lines = f.readlines()
            for line in lines:
                cos = line.split(" ")
                pts.append([int(cos[0]), int(cos[1])])
        predictor = SamPredictor(sam)
        predictor.set_image(cvImage)
        input_point = np.array(pts)
        input_label = np.array([1] * len(input_point))
        input_box = np.array(boxCos) if boxCos else None
        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=(maskType == "Multiple"),
        )
        saveMasks(masks, saveFileNoExt, formatBinary)

    def run_test(self, sam):
        npArr = np.zeros((50, 50), np.uint8)
        cvImage = cv2.cvtColor(npArr, cv2.COLOR_GRAY2BGR)
        predictor = SamPredictor(sam)
        predictor.set_image(cvImage)
        input_box = np.array([10, 10, 20, 20])
        predictor.predict(
            point_coords=None, point_labels=None, box=input_box, multimask_output=False
        )


class SAM2Strategy(SegmentationStrategy):
    MODEL_TYPE_LOOKUP = {
        "sam2_hiera_large": "sam2_hiera_large",
        "sam2_hiera_base_plus": "sam2_hiera_base_plus",
        "sam2_hiera_small": "sam2_hiera_small",
        "sam2_hiera_tiny": "sam2_hiera_tiny",
        "sam2.1_hiera_large": "sam2_hiera_large",
        "sam2.1_hiera_base_plus": "sam2_hiera_base_plus",
        "sam2.1_hiera_small": "sam2_hiera_small",
        "sam2.1_hiera_tiny": "sam2_hiera_tiny",
    }

    def __init__(self):
        self._temp_pth_path = None

    def get_model_type_from_filename(self, model_filename):
        filename_stem = os.path.splitext(model_filename)[0]
        model_type = self.MODEL_TYPE_LOOKUP.get(filename_stem)
        if model_type:
            print(f"Auto-detected SAM2 model type: {model_type}")
            return model_type
        else:
            print(
                f"Error: Could not auto-detect model type from SAM2 filename: {model_filename}"
            )
            print(
                f"Please use one of the following file names (or their .safetensors/.pt equivalents): {list(self.MODEL_TYPE_LOOKUP.keys())}"
            )
            return None

    def _convert_safetensors_to_pth(self, safetensors_path, pth_path):
        try:
            from safetensors.torch import load_file

            state_dict = load_file(safetensors_path)
            checkpoint = {"model": state_dict}
            torch.save(checkpoint, pth_path)
            return True
        except Exception as e:
            print(f"Error converting safetensors to pth: {e}")
            return False

    def load_model(self, checkPtFilePath, modelType):
        model_configs = {
            "sam2_hiera_tiny": "sam2_hiera_t.yaml",
            "sam2_hiera_small": "sam2_hiera_s.yaml",
            "sam2_hiera_base_plus": "sam2_hiera_b+.yaml",
            "sam2_hiera_large": "sam2_hiera_l.yaml",
        }
        config_file = model_configs.get(modelType, "sam2_hiera_l.yaml")
        actual_checkpoint_path = checkPtFilePath
        if checkPtFilePath.endswith(".safetensors"):
            print("Converting safetensors to pth format...")
            self._temp_pth_path = checkPtFilePath.replace(".safetensors", "_temp.pth")
            if self._convert_safetensors_to_pth(checkPtFilePath, self._temp_pth_path):
                actual_checkpoint_path = self._temp_pth_path
                print(f"Converted to: {self._temp_pth_path}")
            else:
                print("Failed to convert safetensors file")
                return None
        try:
            sam = build_sam2(config_file, actual_checkpoint_path)
            print("SAM2 Model loaded successfully!")
            return sam
        except Exception as e:
            print(f"Error loading SAM2 model: {e}")
            self.cleanup()
            return None

    def segment_auto(self, sam, cvImage, saveFileNoExt, formatBinary, **kwargs):
        points_per_side = 32
        if kwargs.get("segRes") == "Low":
            points_per_side = 16
        elif kwargs.get("segRes") == "High":
            points_per_side = 64
        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam,
            points_per_side=points_per_side,
            crop_n_layers=kwargs.get("cropNLayers", 0),
            min_mask_region_area=kwargs.get("minMaskArea", 0),
        )
        masks = mask_generator.generate(cvImage)
        masks = [mask["segmentation"] for mask in masks]
        saveMasks(masks, saveFileNoExt, formatBinary)

    def segment_box(self, sam, cvImage, maskType, boxCos, saveFileNoExt, formatBinary, imgsz=SAM3_DEFAULT_IMGSZ):
        predictor = SAM2ImagePredictor(sam)
        predictor.set_image(cvImage)
        input_box = np.array(boxCos)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box,
            multimask_output=(maskType == "Multiple"),
        )
        saveMasks(masks, saveFileNoExt, formatBinary)

    def segment_sel(
        self, sam, cvImage, maskType, selFile, boxCos, saveFileNoExt, formatBinary, imgsz=SAM3_DEFAULT_IMGSZ
    ):
        pts = []
        with open(selFile, "r") as f:
            lines = f.readlines()
            for line in lines:
                cos = line.split(" ")
                pts.append([int(cos[0]), int(cos[1])])
        predictor = SAM2ImagePredictor(sam)
        predictor.set_image(cvImage)
        input_point = np.array(pts)
        input_label = np.array([1] * len(input_point))
        input_box = np.array(boxCos) if boxCos else None
        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=(maskType == "Multiple"),
        )
        saveMasks(masks, saveFileNoExt, formatBinary)

    def run_test(self, sam):
        npArr = np.zeros((50, 50), np.uint8)
        cvImage = cv2.cvtColor(npArr, cv2.COLOR_GRAY2BGR)
        predictor = SAM2ImagePredictor(sam)
        predictor.set_image(cvImage)
        input_box = np.array([10, 10, 20, 20])
        predictor.predict(
            point_coords=None, point_labels=None, box=input_box, multimask_output=False
        )

    def cleanup(self):
        if self._temp_pth_path and os.path.exists(self._temp_pth_path):
            os.remove(self._temp_pth_path)
            print(f"Removed temporary file: {self._temp_pth_path}")


class SAM3Strategy(SegmentationStrategy):
    MODEL_TYPE_LOOKUP = {"sam3": "sam3"}

    def __init__(self):
        self._semantic_predictor = None
        self._device = None
        self._checkpoint_path = None

    def get_model_type_from_filename(self, model_filename):
        filename_stem = os.path.splitext(model_filename)[0]
        if filename_stem.lower().startswith("sam3"):
            print("Auto-detected SAM3 model type: sam3")
            return "sam3"
        else:
            print(
                f"Error: Could not auto-detect model type from SAM3 filename: {model_filename}"
            )
            return None

    def load_model(self, checkPtFilePath, modelType):
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        self._device = device
        self._checkpoint_path = checkPtFilePath
        try:
            model = UltralyticsSAM(checkPtFilePath)
            print(f"SAM3 Model loaded successfully! (device: {device})")
            return model
        except Exception as e:
            print(f"Error loading SAM3 model: {e}")
            return None

    def _get_semantic_predictor(self, imgsz=SAM3_DEFAULT_IMGSZ):
        if self._semantic_predictor is None:
            overrides = dict(
                conf=0.05,
                task="segment",
                mode="predict",
                imgsz=imgsz,
                model=self._checkpoint_path,
                device=self._device,
                verbose=False,
                save=False,
            )
            self._semantic_predictor = SAM3SemanticPredictor(overrides=overrides)
        return self._semantic_predictor

    def _extract_masks(self, results):
        if not results or results[0].masks is None:
            return []
        return results[0].masks.data.cpu().numpy()

    def _to_bgr(self, cvImage):
        return cv2.cvtColor(cvImage, cv2.COLOR_RGB2BGR)

    def segment_auto(self, sam, cvImage, saveFileNoExt, formatBinary, imgsz=SAM3_DEFAULT_IMGSZ, **kwargs):
        h, w = cvImage.shape[:2]
        logging.info(f"SAM3 Auto: image={w}x{h}, imgsz={imgsz}")
        t0 = time.time()
        predictor = self._get_semantic_predictor(imgsz=imgsz)
        predictor.set_image(self._to_bgr(cvImage))
        results = predictor(text=["object"])
        elapsed = time.time() - t0
        masks = self._extract_masks(results)
        logging.info(f"SAM3 Auto: {len(masks)} mask(s) in {elapsed:.2f}s")
        saveMasks(masks, saveFileNoExt, formatBinary)

    def segment_box(self, sam, cvImage, maskType, boxCos, saveFileNoExt, formatBinary, imgsz=SAM3_DEFAULT_IMGSZ):
        h, w = cvImage.shape[:2]
        logging.info(f"SAM3 Box: image={w}x{h}, imgsz={imgsz}, box={boxCos}, maskType={maskType}")
        t0 = time.time()
        results = sam.predict(
            source=self._to_bgr(cvImage),
            bboxes=boxCos,
            device=self._device,
            imgsz=imgsz,
            verbose=False,
        )
        elapsed = time.time() - t0
        masks = self._extract_masks(results)
        logging.info(f"SAM3 Box: {len(masks)} mask(s) in {elapsed:.2f}s")
        saveMasks(masks, saveFileNoExt, formatBinary)

    def segment_sel(
        self, sam, cvImage, maskType, selFile, boxCos, saveFileNoExt, formatBinary, imgsz=SAM3_DEFAULT_IMGSZ
    ):
        pts = []
        with open(selFile, "r") as f:
            lines = f.readlines()
            for line in lines:
                cos = line.split(" ")
                pts.append([int(cos[0]), int(cos[1])])
        h, w = cvImage.shape[:2]
        logging.info(f"SAM3 Selection: image={w}x{h}, imgsz={imgsz}, points={len(pts)}, maskType={maskType}")
        t0 = time.time()
        results = sam.predict(
            source=self._to_bgr(cvImage),
            points=pts,
            labels=[1] * len(pts),
            device=self._device,
            imgsz=imgsz,
            verbose=False,
        )
        elapsed = time.time() - t0
        masks = self._extract_masks(results)
        logging.info(f"SAM3 Selection: {len(masks)} mask(s) in {elapsed:.2f}s")
        saveMasks(masks, saveFileNoExt, formatBinary)

    def segment_text(self, sam, cvImage, saveFileNoExt, formatBinary, textPrompt, imgsz=SAM3_DEFAULT_IMGSZ):
        prompts = [p.strip() for p in textPrompt.split(",") if p.strip()]
        h, w = cvImage.shape[:2]
        logging.info(f"SAM3 Text: image={w}x{h}, imgsz={imgsz}, prompts={prompts}")
        t0 = time.time()
        predictor = self._get_semantic_predictor(imgsz=imgsz)
        predictor.set_image(self._to_bgr(cvImage))
        results = predictor(text=prompts)
        elapsed = time.time() - t0
        masks = self._extract_masks(results)
        logging.info(f"SAM3 Text: {len(masks)} mask(s) in {elapsed:.2f}s")
        saveMasks(masks, saveFileNoExt, formatBinary)

    def run_test(self, sam):
        npArr = np.zeros((50, 50, 3), np.uint8)
        sam.predict(
            source=npArr, bboxes=[10, 10, 20, 20], device=self._device,
            imgsz=SAM3_DEFAULT_IMGSZ, verbose=False,
        )


def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.info(f"seganybridge args: {sys.argv[1:]}")

    if len(sys.argv) < 3:
        print(
            "Usage: python seganybridge.py <model_type|auto> <checkpoint_path> [options]"
        )
        return

    modelType = sys.argv[1]
    checkPtFilePath = sys.argv[2]
    model_filename = os.path.basename(checkPtFilePath)

    if model_filename.lower().startswith("sam_"):
        strategy = SAM1Strategy()
    elif model_filename.lower().startswith("sam2"):
        strategy = SAM2Strategy()
    elif model_filename.lower().startswith("sam3"):
        if not HAS_SAM3:
            print("Error: sam3 package not installed")
            return
        strategy = SAM3Strategy()
    else:
        print(
            f"Error: Could not determine model family from filename: {model_filename}"
        )
        print(
            "Filename must start with 'sam_' for SAM1, 'sam2' for SAM2, or 'sam3' for SAM3."
        )
        return

    if modelType.lower() == "auto":
        modelType = strategy.get_model_type_from_filename(model_filename)
        if not modelType:
            return

    if not os.path.exists(checkPtFilePath):
        print(f"Error: Checkpoint file not found: {checkPtFilePath}")
        return

    t0 = time.time()
    sam = strategy.load_model(checkPtFilePath, modelType)
    if sam is None:
        return
    logging.info(f"Model loaded in {time.time() - t0:.2f}s")

    if not isinstance(strategy, SAM3Strategy):
        if torch.cuda.is_available():
            sam.to(device="cuda")
            print("Model moved to CUDA")

    if len(sys.argv) == 3:
        strategy.run_test(sam)
        print("Success!!")
        strategy.cleanup()
        return

    ipFile = sys.argv[3]
    segType = sys.argv[4]
    maskType = sys.argv[5]
    saveFileNoExt = sys.argv[6]
    formatBinary = sys.argv[7] == "True" if len(sys.argv) > 7 else True

    cvImage = cv2.imread(ipFile)
    cvImage = cv2.cvtColor(cvImage, cv2.COLOR_BGR2RGB)
    h, w = cvImage.shape[:2]
    logging.info(f"Input image: {ipFile} ({w}x{h})")

    global _orig_image_size
    if max(h, w) > MAX_INPUT_DIM:
        scale = MAX_INPUT_DIM / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        _orig_image_size = (h, w)
        cvImage = cv2.resize(cvImage, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logging.info(f"Downscaled to {new_w}x{new_h} (masks will be upscaled back to {w}x{h})")

    try:
        if segType == "Auto":
            auto_kwargs = {}
            imgsz = SAM3_DEFAULT_IMGSZ
            if isinstance(strategy, SAM3Strategy):
                if len(sys.argv) > 8:
                    imgsz = int(sys.argv[8])
            elif isinstance(strategy, SAM2Strategy):
                if len(sys.argv) > 8:
                    auto_kwargs["segRes"] = sys.argv[8]
                if len(sys.argv) > 9:
                    auto_kwargs["cropNLayers"] = int(sys.argv[9])
                if len(sys.argv) > 10:
                    auto_kwargs["minMaskArea"] = int(sys.argv[10])
            strategy.segment_auto(
                sam, cvImage, saveFileNoExt, formatBinary, imgsz, **auto_kwargs
            )
        elif segType in {"Selection", "Box-Selection"}:
            selFile = sys.argv[8]
            boxCos = (
                [float(val.strip()) for val in sys.argv[9].split(",")]
                if len(sys.argv) > 9
                else None
            )
            strategy.segment_sel(
                sam, cvImage, maskType, selFile, boxCos, saveFileNoExt, formatBinary
            )
        elif segType == "Box":
            boxCos = [float(val.strip()) for val in sys.argv[9].split(",")]
            strategy.segment_box(
                sam, cvImage, maskType, boxCos, saveFileNoExt, formatBinary
            )
        elif segType == "Text":
            textPrompt = sys.argv[8]
            imgsz = int(sys.argv[9]) if len(sys.argv) > 9 else SAM3_DEFAULT_IMGSZ
            strategy.segment_text(
                sam, cvImage, saveFileNoExt, formatBinary, textPrompt, imgsz
            )
        else:
            print(f"Unknown segmentation type: {segType}")
    finally:
        print("Done!")
        strategy.cleanup()


if __name__ == "__main__":
    main()

