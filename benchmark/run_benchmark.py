import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", help="Experiment number to run (comma separated list, e.g. --test=00,02)", default='')
    parser.add_argument("--train", help="Training experiment number to run (comma separated list, e.g. --train=noaug)", default='')
    parser.add_argument("--score", help="Scoring only during test", action='store_true')
    parser.add_argument("--takeac", help="Number of elements to take for AC task", default=13000)
    parser.add_argument("--takead", help="Number of elements to take for AD task", default=None)
    parser.add_argument("--outdir", help="Output directory", default='./cap_benchmark')
    args = parser.parse_args()

    scoreflag = '--score' if args.score else ''
    if '00' in args.test:
        cmd = 'python -u  cap_benchmark.py --model=cap_classification_pad --dataset=cap_classification_pad --test %s --confusion --take=%s --outdir=%s' % (scoreflag, str(args.takeac), args.outdir); print('[run_benchmark]: %s' % cmd); os.system(cmd)        
    if '01' in args.test:
        cmd = 'python -u  cap_benchmark.py --model=cap_classification_pad --dataset=cap_classification_pad_stabilized --test %s --confusion --take=%s --outdir=%s' % (scoreflag, str(args.takeac), args.outdir); print('[run_benchmark]: %s' % cmd); os.system(cmd)
    if '02' in args.test:
        cmd = 'python -u  cap_benchmark.py --model=cap_classification_pad --dataset=cap_detection_handheld --test %s --take=%s --outdir=%s ' % (scoreflag, str(args.takead), args.outdir); print('[run_benchmark]: %s' % cmd); os.system(cmd)
    if '03' in args.test:
        cmd = 'python -u  cap_benchmark.py --model=cap_classification_pad --dataset=cap_detection_rigid --test %s --take=%s --outdir=%s ' % (scoreflag, str(args.takead), args.outdir); print('[run_benchmark]: %s' % cmd); os.system(cmd)

    if '10' in args.test:
        cmd = 'python -u  cap_benchmark.py --model=cap_classification_pad_stabilized --dataset=cap_classification_pad --test %s --confusion --take=%s --outdir=%s ' % (scoreflag, str(args.takeac), args.outdir); print('[run_benchmark]: %s' % cmd); os.system(cmd)
    if '11' in args.test:
        cmd = 'python -u  cap_benchmark.py --model=cap_classification_pad_stabilized --dataset=cap_classification_pad_stabilized --test %s --confusion --take=%s --outdir=%s ' % (scoreflag, str(args.takeac), args.outdir); print('[run_benchmark]: %s' % cmd); os.system(cmd)
    if '12' in args.test:
        cmd = 'python -u  cap_benchmark.py --model=cap_classification_pad_stabilized --dataset=cap_detection_handheld --test %s --take=%s --outdir=%s ' % (scoreflag, str(args.takead), args.outdir); print('[run_benchmark]: %s' % cmd); os.system(cmd)
    if '13' in args.test:
        cmd = 'python -u  cap_benchmark.py --model=cap_classification_pad_stabilized --dataset=cap_detection_rigid --test %s --take=%s --outdir=%s ' % (scoreflag, str(args.takead), args.outdir); print('[run_benchmark]: %s' % cmd); os.system(cmd)

    if '20' in args.test:
        cmd = 'python -u  cap_benchmark.py --model=cap_classification_pad_coarsened --dataset=cap_classification_pad_coarse --test %s --take=%s --outdir=%s ' % (scoreflag, str(args.takeac), args.outdir); print('[run_benchmark]: %s' % cmd); os.system(cmd)
    if '21' in args.test:
        cmd = 'python -u  cap_benchmark.py --model=cap_classification_pad_coarsened --dataset=cap_classification_pad_stabilized_coarse --test %s --take=%s --outdir=%s ' % (scoreflag, str(args.takeac), args.outdir); print('[run_benchmark]: %s' % cmd); os.system(cmd)
    if '22' in args.test:
        cmd = 'python -u  cap_benchmark.py --model=cap_classification_pad_coarsened --dataset=cap_detection_handheld_coarse --test %s --take=%s --outdir=%s ' % (scoreflag, str(args.takead), args.outdir); print('[run_benchmark]: %s' % cmd); os.system(cmd)
    if '23' in args.test:
        cmd = 'python -u  cap_benchmark.py --model=cap_classification_pad_coarsened  --dataset=cap_detection_rigid_coarse --test %s --take=%s --outdir=%s ' % (scoreflag, str(args.takead), args.outdir); print('[run_benchmark]: %s' % cmd); os.system(cmd)

    if '30' in args.test:
        cmd = 'python -u  cap_benchmark.py --model=cap_classification_pad_stabilized_coarsened --dataset=cap_classification_pad_coarse --test %s --take=%s --outdir=%s' % (scoreflag, str(args.takeac), args.outdir); print('[run_benchmark]: %s' % cmd); os.system(cmd)
    if '31' in args.test:
        cmd = 'python -u  cap_benchmark.py --model=cap_classification_pad_stabilized_coarsened --dataset=cap_classification_pad_stabilized_coarse  --test %s --take=%s --outdir=%s' % (scoreflag, str(args.takeac), args.outdir); print('[run_benchmark]: %s' % cmd); os.system(cmd)
    if '32' in args.test:
        cmd = 'python -u  cap_benchmark.py --model=cap_classification_pad_stabilized_coarsened --dataset=cap_detection_handheld_coarse  --test %s --take=%s --outdir=%s ' % (scoreflag, str(args.takead), args.outdir); print('[run_benchmark]: %s' % cmd); os.system(cmd)
    if '33' in args.test:
        cmd = 'python -u  cap_benchmark.py --model=cap_classification_pad_stabilized_coarsened --dataset=cap_detection_rigid_coarse  --test %s --take=%s --outdir=%s ' % (scoreflag, str(args.takead), args.outdir); print('[run_benchmark]: %s' % cmd); os.system(cmd)

    if '40' in args.test:
        cmd = 'python -u  cap_benchmark.py --model=cap_classification_pad_coarse --dataset=cap_classification_pad_coarse --test %s --take=%s --outdir=%s' % (scoreflag, str(args.takeac), args.outdir); print('[run_benchmark]: %s' % cmd); os.system(cmd)
    if '41' in args.test:
        cmd = 'python -u  cap_benchmark.py --model=cap_classification_pad_coarse --dataset=cap_classification_pad_stabilized_coarse  --test %s --take=%s --outdir=%s' % (scoreflag, str(args.takeac), args.outdir); print('[run_benchmark]: %s' % cmd); os.system(cmd)
    if '42' in args.test:
        cmd = 'python -u  cap_benchmark.py --model=cap_classification_pad_coarse --dataset=cap_detection_handheld_coarse  --test %s --take=%s --outdir=%s ' % (scoreflag, str(args.takead), args.outdir); print('[run_benchmark]: %s' % cmd); os.system(cmd)
    if '43' in args.test:
        cmd = 'python -u  cap_benchmark.py --model=cap_classification_pad_coarse --dataset=cap_detection_rigid_coarse  --test %s --take=%s --outdir=%s ' % (scoreflag, str(args.takead), args.outdir); print('[run_benchmark]: %s' % cmd); os.system(cmd)

    if '50' in args.test:
        cmd = 'python -u  cap_benchmark.py --model=cap_classification_pad_stabilized_coarse --dataset=cap_classification_pad_coarse --test %s --take=%s --outdir=%s' % (scoreflag, str(args.takeac), args.outdir); print('[run_benchmark]: %s' % cmd); os.system(cmd)
    if '51' in args.test:
        cmd = 'python -u  cap_benchmark.py --model=cap_classification_pad_stabilized_coarse --dataset=cap_classification_pad_stabilized_coarse  --test %s --take=%s --outdir=%s' % (scoreflag, str(args.takeac), args.outdir); print('[run_benchmark]: %s' % cmd); os.system(cmd)
    if '52' in args.test:
        cmd = 'python -u  cap_benchmark.py --model=cap_classification_pad_stabilized_coarse --dataset=cap_detection_handheld_coarse  --test %s --take=%s --outdir=%s ' % (scoreflag, str(args.takead), args.outdir); print('[run_benchmark]: %s' % cmd); os.system(cmd)
    if '53' in args.test:
        cmd = 'python -u  cap_benchmark.py --model=cap_classification_pad_stabilized_coarse --dataset=cap_detection_rigid_coarse  --test %s --take=%s --outdir=%s ' % (scoreflag, str(args.takead), args.outdir); print('[run_benchmark]: %s' % cmd); os.system(cmd)

    if 'noaug' in args.train:
        cmd = 'PL_TORCH_DISTRIBUTED_BACKEND=gloo python -u  cap_benchmark.py --model=cap_classification_pad --dataset=cap_classification_pad  --train --noaug --outdir=%s | tee %s/cap_benchmark_noaug.log' % (args.outdir, args.outdir); print('[run_benchmark]: %s' % cmd); os.system(cmd)
    if 'noaway' in args.train:
        cmd = 'PL_TORCH_DISTRIBUTED_BACKEND=gloo python -u  cap_benchmark.py --model=cap_classification_pad --dataset=cap_classification_pad  --train --noaway --outdir=%s | tee %s/cap_benchmark_noaway.log' % (args.outdir, args.outdir); print('[run_benchmark]: %s' % cmd); os.system(cmd)
    if 'coarse_stabilized' in args.train:
        cmd = 'PL_TORCH_DISTRIBUTED_BACKEND=gloo python -u  cap_benchmark.py --model=cap_classification_pad_stabilized --dataset=cap_classification_pad_stabilized  --train --coarse --outdir=%s | tee %s/cap_benchmark_coarse_stabilized.log' % (args.outdir, args.outdir); print('[run_benchmark]: %s' % cmd); os.system(cmd)
    elif 'coarse' in args.train:
        cmd = 'PL_TORCH_DISTRIBUTED_BACKEND=gloo python -u  cap_benchmark.py --model=cap_classification_pad --dataset=cap_classification_pad  --train --coarse --outdir=%s | tee %s/cap_benchmark_coarse.log' % (args.outdir, args.outdir); print('[run_benchmark]: %s' % cmd); os.system(cmd)
    if 'supercollector' in args.train:
        cmd = 'PL_TORCH_DISTRIBUTED_BACKEND=gloo python -u  cap_benchmark.py --model=cap_classification_pad --dataset=cap_classification_pad  --train --supercollector --outdir=%s | tee %s/cap_benchmark_supercollector.log' % (args.outdir, args.outdir); print('[run_benchmark]: %s' % cmd); os.system(cmd)
        
    if 'supercollector' in args.test:
        cmd = 'python -u  cap_benchmark.py --model=cap_classification_pad_supercollector --dataset=cap_classification_pad  --test %s --take=%s --outdir=%s ' % (scoreflag, str(args.takeac), args.outdir); print('[run_benchmark]: %s' % cmd); os.system(cmd)
    if 'noaway' in args.test:
        cmd = 'python -u  cap_benchmark.py --model=cap_classification_pad_noaway --dataset=cap_classification_pad  --test %s --take=%s --outdir=%s ' % (scoreflag, str(args.takeac), args.outdir); print('[run_benchmark]: %s' % cmd); os.system(cmd)
    if 'noaug' in args.test:
        cmd = 'python -u  cap_benchmark.py --model=cap_classification_pad_noaug --dataset=cap_classification_pad  --test %s --take=%s --outdir=%s ' % (scoreflag, str(args.takeac), args.outdir); print('[run_benchmark]: %s' % cmd); os.system(cmd) 
        


        
