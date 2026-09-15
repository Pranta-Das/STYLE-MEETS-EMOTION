import os
import shutil
from pathlib import Path

def collect_images(src_dirs, dst_dir, exts=('jpg','jpeg','png')):
    os.makedirs(dst_dir, exist_ok=True)
    count = 0
    for src in src_dirs:
        for p in Path(src).rglob('*'):
            if p.suffix.lower().lstrip('.') in exts:
                dst = Path(dst_dir) / p.name
                if not dst.exists():
                    try:
                        shutil.copy(p, dst)
                        count += 1
                    except Exception:
                        continue
    return count

def main():
    # default layout
    natural_src = os.environ.get('NATURAL_SRC', './ffhq-dataset/images1024x1024')
    artistic_src = os.environ.get('AAHQ_SRC', './aahq-dataset/aligned')
    out_base = os.environ.get('OUT_BASE', './data')

    natural_dst = os.path.join(out_base, 'natural_faces')
    artistic_dst = os.path.join(out_base, 'artistic_styles')

    print('Collecting natural faces...')
    n = collect_images([natural_src], natural_dst)
    print(f'Copied {n} natural face images to {natural_dst}')

    print('Collecting artistic style images...')
    m = collect_images([artistic_src], artistic_dst)
    print(f'Copied {m} artistic images to {artistic_dst}')

    print('Done. You can now point style_image_paths to:', natural_dst, artistic_dst)

if __name__ == '__main__':
    main()
