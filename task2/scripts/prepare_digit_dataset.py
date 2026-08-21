"""Convert RMData2 folders to an Ultralytics classification dataset."""
import argparse
import shutil
from pathlib import Path

DIGITS = {'1', '2', '3', '4', '5'}


def target_class(name: str) -> str:
    return name if name in DIGITS else 'unknown'


def prepare(source: Path, output: Path) -> None:
    split_map = {'update_train': 'train', 'update_test': 'val'}
    for source_split, output_split in split_map.items():
        for class_dir in (source / source_split).iterdir():
            if not class_dir.is_dir():
                continue
            target = output / output_split / target_class(class_dir.name)
            target.mkdir(parents=True, exist_ok=True)
            for image in class_dir.glob('*.png'):
                shutil.copy2(image, target / f'{class_dir.name}_{image.name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    prepare(args.source, args.output)
