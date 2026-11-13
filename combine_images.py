import os, shutil, argparse

def combine(src1, src2, dest):
    os.makedirs(dest, exist_ok=True)
    for src in (src1, src2):
        if not os.path.isdir(src):
            print("Warning: source not found:", src)
            continue
        for fname in os.listdir(src):
            srcf = os.path.join(src, fname)
            destf = os.path.join(dest, fname)
            if os.path.isfile(srcf):
                # avoid overwriting
                if os.path.exists(destf):
                    print("Skipping existing:", destf)
                else:
                    shutil.copy2(srcf, destf)
    print("Done combining into", dest)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--src1', required=True)
    p.add_argument('--src2', required=True)
    p.add_argument('--dest', default='dataset/combined_images')
    args = p.parse_args()
    combine(args.src1, args.src2, args.dest)
