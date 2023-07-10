import os
import random
import hashlib
import argparse 
import vipy
import vipy.batch
import vipy.dataset
import vipy.torch
import vipy.show
import heyvi.recognition
import heyvi.detection
import torch
import pytorch_lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin             
import torch.utils.data
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
import gc 
import h5py
import matplotlib.pyplot as plt


# Update these paths to point to your local directories:
# - The naming convention "cap_classification_pad.fbdc75e6ef10b874ddda20ee9765a710" is a tarball cap_classification_pad.tar.gz with MD5 sum "fbdc75e6ef10b874ddda20ee9765a710".
# - Unpack the tarball, and rename the directory it unpacks to using this convention
# - The files in directories appended with "_truth" are sequestered and are not used by leaderboard submitters
# - The files in directories appended with "_test" are available for download on the leaderboard after submitting a license agreement
DATASET = {'cap_classification_pad':{'trainpath':'cap_classification_pad.fbdc75e6ef10b874ddda20ee9765a710',
                                     'trainsplit':'cap_classification_pad.fbdc75e6ef10b874ddda20ee9765a710/train_val.json',
                                     'testpath':'cap_classification_pad_test.c241c3d0200128a0540e4e15d831954f',
                                     'testindex':'cap_classification_pad_test.c241c3d0200128a0540e4e15d831954f/cap_activity_classification_index.csv',
                                     'testjson':'cap_classification_pad_test.c241c3d0200128a0540e4e15d831954f/cap_classification_pad_test.json',
                                     'valref':None,
                                     'testref':'cap_classification_pad_test_truth.dc5c90bfe43fadc7ddd46a11d90b2dbb/cap_activity_classification_ref_openfad.csv',
                                     'truthjson':'cap_classification_pad_test_truth.dc5c90bfe43fadc7ddd46a11d90b2dbb/annotations',
                                     'truthsplit':'cap_classification_pad_test_truth.dc5c90bfe43fadc7ddd46a11d90b2dbb/train_val_test.json',
                                     'truthiid':'cap_classification_pad_test_truth.dc5c90bfe43fadc7ddd46a11d90b2dbb/test_instanceid_to_ref_instanceid.json'},

           'cap_classification_pad_coarse':{'trainpath':None,
                                            'trainsplit':None,
                                            'testpath':'cap_classification_pad_test.c241c3d0200128a0540e4e15d831954f',
                                            'testindex':'cap_classification_pad_test.c241c3d0200128a0540e4e15d831954f/cap_activity_classification_index.csv',
                                            'testjson':'cap_classification_pad_test.c241c3d0200128a0540e4e15d831954f/cap_classification_pad_test.json',
                                            'valref':None,
                                            'testref':'cap_classification_pad_coarse_ref.csv',
                                            'truthjson':None,
                                            'truthsplit':None,
                                            'truthiid':'cap_classification_pad_test_truth.dc5c90bfe43fadc7ddd46a11d90b2dbb/test_instanceid_to_ref_instanceid.json'},

           'cap_detection_handheld':{'trainpath':None,
                                     'trainsplit':None,
                                     'testpath':'cap_detection_handheld.bd4e3a870b95324e2a4c0204bf4d34a4',
                                     'testindex':'cap_detection_handheld.bd4e3a870b95324e2a4c0204bf4d34a4/cap_detection_handheld_index.csv',
                                     'testjson':'cap_detection_handheld.bd4e3a870b95324e2a4c0204bf4d34a4/annotations/CAP Detection.json',
                                     'valref':None,
                                     'testref':'cap_detection_handheld_truth.2b50f8ab0ca52588c189309cb876f120/cap_detection_handheld_ref.csv',
                                     'truthjson':'cap_detection_handheld_truth.2b50f8ab0ca52588c189309cb876f120/annotations',
                                     'truthsplit':None,
                                     'truthiid':None},

           'cap_detection_handheld_coarse':{'trainpath':None,
                                            'trainsplit':None,
                                            'testpath':'cap_detection_handheld.bd4e3a870b95324e2a4c0204bf4d34a4',
                                            'testindex':'cap_detection_handheld.bd4e3a870b95324e2a4c0204bf4d34a4/cap_detection_handheld_index.csv',
                                            'testjson':'cap_detection_handheld.bd4e3a870b95324e2a4c0204bf4d34a4/annotations/CAP Detection.json',
                                            'valref':None,
                                            'testref':'cap_detection_handheld_coarse_ref.csv',
                                            'truthjson':'cap_detection_handheld_truth.2b50f8ab0ca52588c189309cb876f120/annotations',
                                            'truthsplit':None,
                                            'truthiid':None},

           'cap_detection_rigid':{'trainpath':None,
                                       'trainsplit':None,
                                       'testpath':'cap_detection_stabilized.f36361e769b6da7cb12089d5211ff3b3',
                                       'testindex':'cap_detection_stabilized.f36361e769b6da7cb12089d5211ff3b3/cap_detection_stabilized_index.csv',
                                       'testjson':'cap_detection_stabilized.f36361e769b6da7cb12089d5211ff3b3/annotations/CAP Detection.json',
                                       'valref':None,
                                       'testref':'cap_detection_stabilized_truth.1b5e9b7f2a9f96c7dd0405345d3e123d/cap_detection_stabilized_ref.csv',
                                       'truthjson':'cap_detection_stabilized_truth.1b5e9b7f2a9f96c7dd0405345d3e123d/annotations',
                                       'truthsplit':None,
                                       'truthiid':None},

           'cap_detection_rigid_coarse':{'trainpath':None,
                                         'trainsplit':None,
                                         'testpath':'cap_detection_stabilized.f36361e769b6da7cb12089d5211ff3b3',
                                         'testindex':'cap_detection_stabilized.f36361e769b6da7cb12089d5211ff3b3/cap_detection_stabilized_index.csv',
                                         'testjson':'cap_detection_stabilized.f36361e769b6da7cb12089d5211ff3b3/annotations/CAP Detection.json',
                                         'valref':None,
                                         'testref':'cap_detection_rigid_coarse_ref.csv',
                                         'truthjson':'cap_detection_stabilized_truth.1b5e9b7f2a9f96c7dd0405345d3e123d/annotations',
                                         'truthsplit':None,
                                         'truthiid':None},

           'cap_classification_pad_stabilized':{'trainpath':'cap_classification_pad_stabilize.b8276e697226988d312999dd6797f42e',
                                                'trainsplit':'cap_classification_pad_stabilize.b8276e697226988d312999dd6797f42e/train_val.json',
                                                'testpath':'cap_classification_pad_stabilize_test.032f36f78f8963f69430c4c4af9803c0',
                                                'testindex':'cap_classification_pad_stabilize_test.032f36f78f8963f69430c4c4af9803c0/cap_activity_classification_index.csv',
                                                'testjson':'cap_classification_pad_stabilize_test.032f36f78f8963f69430c4c4af9803c0/cap_classification_pad_stabilize_test.json',
                                                'valref':None,
                                                'testref':'cap_classification_pad_stabilize_test_truth.03aae4b6fc57f54409098a25968c4146/cap_activity_classification_ref_openfad.csv',
                                                'truthjson':None,
                                                'truthsplit':'cap_classification_pad_stabilize_test_truth.03aae4b6fc57f54409098a25968c4146/train_val_test.json',
                                                'truthiid':'cap_classification_pad_stabilize_test_truth.03aae4b6fc57f54409098a25968c4146/test_instanceid_to_ref_instanceid.json'},
           
           'cap_classification_pad_stabilized_coarse':{'trainpath':None,
                                                       'trainsplit':'cap_classification_pad.fbdc75e6ef10b874ddda20ee9765a710/train_val.json',
                                                       'testpath':'cap_classification_pad_stabilize_test.032f36f78f8963f69430c4c4af9803c0',
                                                       'testindex':'cap_classification_pad_stabilize_test.032f36f78f8963f69430c4c4af9803c0/cap_activity_classification_index.csv',
                                                       'testjson':'cap_classification_pad_stabilize_test.032f36f78f8963f69430c4c4af9803c0/cap_classification_pad_stabilize_test.json',
                                                       'valref':None,
                                                       'testref':'cap_classification_pad_stabilized_coarse_ref.csv',
                                                       'truthjson':None,
                                                       'truthsplit':None,
                                                       'truthiid':'cap_classification_pad_stabilize_test_truth.03aae4b6fc57f54409098a25968c4146/test_instanceid_to_ref_instanceid.json'}}

MODEL = {'cap_classification_pad':{'modelpath':'cap_benchmark/cap_classification_pad_e18s57740.ckpt',
                                   'labelset':'cap'},
         'cap_classification_pad_supercollector':{'modelpath':'cap_benchmark/cap_classification_pad_supercollector_e0s59692v3.ckpt',
                                                  'labelset':'cap'},
         'cap_classification_pad_noaway':{'modelpath':'cap_benchmark/cap_classification_pad_noaway_e1s62213v1.ckpt',
                                          'labelset':'cap'},
         'cap_classification_pad_noaug':{'modelpath':'cap_benchmark/cap_classification_pad_noaug_e5s75975v5.ckpt',
                                          'labelset':'cap'},
         'cap_classification_pad_stabilized':{'modelpath':'cap_benchmark/cap_classification_pad_stabilized_e18s56790v8.ckpt',
                                              'labelset':'cap_stabilized'},
         'cap_classification_pad_stabilized_coarsened':{'modelpath':'cap_benchmark/cap_classification_pad_stabilized_e18s56790v8.ckpt',
                                                        'labelset':'cap_coarsened'},
         'cap_classification_pad_coarsened':{'modelpath':'cap_benchmark/cap_classification_pad_e18s57740.ckpt',
                                             'labelset':'cap_coarsened'},
         'cap_classification_pad_coarse':{'modelpath':'cap_benchmark/cap_classification_pad_coarselabel_e28s145872v1.ckpt',
                                          'labelset':'cap_coarse_v2'},
         'cap_classification_pad_stabilized_coarse':{'modelpath':'cap_benchmark/cap_classification_pad_stabilized_coarselabel_e24s131516v0.ckpt',
                                          'labelset':'cap_coarse_v2'},
         'cap_detection_handheld_coarse':{'modelpath':'/path/to/out.ckpt',
                                          'labelset':'cap_coarse'}}



def trainvalset(name):
    assert name in DATASET
    d = vipy.util.load(DATASET[name]['trainsplit'])
    (trainid, valid) = (set(d['train']), set(d['val']))
    D = vipy.dataset.Dataset([v for f in vipy.util.findjson(os.path.join(DATASET[name]['trainpath'], 'annotations')) for v in vipy.util.load(f)], id=name)
    (trainset, valset) = (vipy.dataset.Dataset([v for v in D if v.instanceid() in trainid], id=D.id()), vipy.dataset.Dataset([v for v in D if v.instanceid() in valid], id=D.id()))    
    return (trainset, valset)


def trainset(name):
    (trainset, valset) = trainvalset(name)
    return trainset


def valset(name):
    (trainset, valset) = trainvalset(name)
    return valset


def testset(name):
    assert name in DATASET
    testid = set([iid for (iid, framerate) in vipy.util.readcsv(DATASET[name]['testindex'], comment='#')])
    D = vipy.dataset.Dataset(vipy.util.load(DATASET[name]['testjson']), id=name)
    return D.filter(lambda v: v.instanceid() in testid)

def testtruth(name):
    assert name in DATASET
    return vipy.dataset.Dataset([v for f in vipy.util.findjson(DATASET[name]['truthjson']) for v in vipy.util.load(f)], id=name)
    

def otherset(outdir):
    """Derived datasets for attributes under test"""

    d_label_to_superlabel = vipy.util.readjson('cap_label_to_superlabel.json')
    testcsv = vipy.util.readcsv(DATASET['cap_classification_pad']['testref'], ignoreheader=True)
    badlabel = set([c for (iid, c) in testcsv if c not in d_label_to_superlabel])
    print('[cap_benchmark.otherset]: filtering unknown labels: %s' % str(badlabel))
    testcsv = [(iid, d_label_to_superlabel[c]) for (iid, c) in testcsv if c not in badlabel]  # filtered  
    print(vipy.util.writecsv(testcsv, os.path.join(outdir, 'cap_classification_pad_coarse_ref.csv'), header=('video_file_id', 'activity_id')))

    testcsv = vipy.util.readcsv(DATASET['cap_classification_pad_stabilized']['testref'], ignoreheader=True)
    badlabel = set([c for (iid, c) in testcsv if c not in d_label_to_superlabel])
    print('[cap_benchmark.otherset]: filtering unknown labels: %s' % str(badlabel))
    testcsv = [(iid, d_label_to_superlabel[c]) for (iid, c) in testcsv if c not in badlabel]  # filtered  
    print(vipy.util.writecsv(testcsv, os.path.join(outdir, 'cap_classification_pad_stabilized_coarse_ref.csv'), header=('video_file_id', 'activity_id')))

    V = [v for f in vipy.util.findjson(DATASET['cap_detection_handheld']['truthjson']) for v in vipy.util.load(f)]
    d = {v.videoid():v for v in V}
    testid = set([v.videoid() for v in V if 'stabilized' in v.attributes['collection_name']])
    print(vipy.util.writecsv([(a,d_label_to_superlabel[b] if b in d_label_to_superlabel else b,c,d) for (a,b,c,d) in vipy.util.readcsv(DATASET['cap_detection_handheld']['testref'], ignoreheader=True)], 'cap_detection_handheld_coarse_ref.csv', header=vipy.util.readcsv(DATASET['cap_detection_handheld']['testref'])[0]))
    print(vipy.util.writecsv([(a,d_label_to_superlabel[b] if b in d_label_to_superlabel else b,c,d) for (a,b,c,d) in vipy.util.readcsv(DATASET['cap_detection_rigid']['testref'], ignoreheader=True)], 'cap_detection_rigid_coarse_ref.csv', header=vipy.util.readcsv(DATASET['cap_detection_rigid']['testref'])[0]))


def training_weights(name, outdir='/proj/diva3/visym/heyvi/heyvi/model/cap'):
    (trainset, valset) = trainvalset(name)
    vipy.util.writecsv([(k,v) for (k,v) in trainset.union(valset).class_to_index().items()], os.path.join(outdir, '%s_class.csv' % trainset.id()))
    W = trainset.multilabel_inverse_frequency_weight()  # cap inverse frequency weights        
    vipy.util.writecsv(W.items(), os.path.join(outdir, '%s_training_weight.csv' % trainset.id()))
    

def geolocation(name):
    import pycollector.admin.video
    
    (test, (train, val)) = (testtruth(name), trainvalset(name))
    d_collectorid_to_video = {k:vipy.util.takeone(v) for (k,v) in vipy.util.groupbyasdict(train.union(val.union(test)).filter(lambda v: v.getattribute('collector_id') is not None).tolist(), lambda v: v.getattribute('collector_id')).items()}
    d_collectorid_to_geolocation = {k:pycollector.admin.video.Video.cast(v).print(sleep=8).geolocation() for (k,v) in d_collectorid_to_video.items()}
    trainid = set([v.attributes['collector_id'] for v in train] + [v.attributes['collector_id'] for v in val])
    testid = set([v.attributes['collector_id'] for v in test]) 
    print(vipy.util.save((d_collectorid_to_geolocation, list(trainid), list(testid)), 'collectorid_to_geolocation.json'))


def ac_bycountry():    
    testset = testtruth('cap_classification_pad')
    (train, val) = trainvalset('cap_classification_pad')
    (d_collectorid_to_geolocation, traincid, testcid) = vipy.util.readjson('collectorid_to_geolocation.json')
    d_country_to_numcollectors = vipy.util.countby(testcid, lambda t: d_collectorid_to_geolocation[t]['country_name'] if d_collectorid_to_geolocation[t] is not None else 'None')
    d_country_to_numinstances = vipy.util.countby(testset, lambda v: d_collectorid_to_geolocation[v.attributes['collector_id']]['country_name'] if d_collectorid_to_geolocation[v.attributes['collector_id']] is not None else 'None')

    d_country_to_numtrain = vipy.util.countby(train.union(val), lambda v: d_collectorid_to_geolocation[v.attributes['collector_id']]['country_name'] if d_collectorid_to_geolocation[v.attributes['collector_id']] is not None else 'None')
    countryset = set([x for x in d_country_to_numcollectors.keys() if x != 'None'])
    topk_country = [y[0] for y in sorted(list(d_country_to_numinstances.items()), key=lambda x: x[1], reverse=True)[:8] if y[0] != 'None']
    d_iid_to_country = {v.instanceid():d_collectorid_to_geolocation[v.attributes['collector_id']]['country_name'] if (v.attributes['collector_id'] in d_collectorid_to_geolocation and d_collectorid_to_geolocation[v.attributes['collector_id']] is not None) else None for v in testset.union(train.union(val))}
    
    print(sorted(d_country_to_numinstances.items(), key=lambda x: x[1]))
    print(sorted(d_country_to_numtrain.items(), key=lambda x: x[1]))

    scorecsv = vipy.util.readcsv('./cap_benchmark/cap_classification_pad/cap_classification_pad/cap_classification_pad_classification_output.csv')
    refcsv = vipy.util.readcsv(DATASET['cap_classification_pad']['testref'])
    outdir = vipy.util.remkdir('./cap_benchmark/cap_classification_pad/cap_classification_pad/ac_bycountry')
    iid = set([x[0] for x in scorecsv[1:]]).intersection(set([x[0] for x in refcsv[1:]]))

    d_country_to_pr = {c:{} for c in countryset}
    for c in topk_country:
        r = vipy.util.writecsv(refcsv[:1] + [x for x in refcsv[1:] if x[0] in iid and d_iid_to_country[x[0]] == c], os.path.join(outdir, '%s_ref.csv' % c.replace(' ','_')))
        y = vipy.util.writecsv(scorecsv[:1] + [x for x in scorecsv[1:] if x[0] in iid and d_iid_to_country[x[0]] == c], os.path.join(outdir, '%s_score.csv' % c.replace(' ','_')))
        o = vipy.util.remkdir(os.path.join(outdir, c.replace(' ','_')))
        cmd = 'fad-scorer --verbose score-ac -r %s -y %s -o %s' % (r, y, o)
        print(cmd); os.system(cmd)

        f = h5py.File(os.path.join(o, 'scoring_results.h5'))
        d_country_to_pr[c]['precision'] = list(f['system']['prs']['precision'])
        d_country_to_pr[c]['recall'] = list(f['system']['prs']['recall'])

        vipy.metrics.plot_pr(d_country_to_pr[c]['precision'], d_country_to_pr[c]['recall'], label=c, figure=1, outfile=os.path.join(o, 'ac_bycountry.pdf'), fontsize=9)



def tensorset(D, outdir, n_augmentations=4):    
    assert isinstance(D, vipy.dataset.Dataset)
    with vipy.globals.parallel(scheduler='ma01-5200-0044:8785'):
        tensordir = vipy.util.remkdir(os.path.join(outdir, D.id(), 'tensorset'))
        f_instance_to_labeled_tensor = heyvi.recognition.CAP().totensor(training=True)  # create (tensor, label) pairs with data augmentation
        D.to_torch_tensordir(f_instance_to_labeled_tensor, tensordir, n_augmentations=n_augmentations, sleep=None)
    return tensordir


def train(trainsetid, valsetid, tensordir, outdir, batchsize=32, num_workers=4, resume_from_checkpoint=None, version=7):
    gc.disable()

    # 8-GPU machine training
    assert heyvi.version.is_at_least('0.2.26')  # originally trained on heyvi-0.2.25, updated labelset kwarg for 0.2.26
    net = heyvi.recognition.CAP(labelset=version)  
    checkpoint_callback = ModelCheckpoint(save_top_k=5, monitor='avg_val_loss', verbose=True, mode='min')  
    t = pl.Trainer(gpus=[0,1,2,3,4,5,6,7], accelerator='ddp', default_root_dir=outdir, resume_from_checkpoint=resume_from_checkpoint, plugins=DDPPlugin(find_unused_parameters=False), callbacks=[checkpoint_callback])

    # Datasets 
    traintensor = vipy.torch.Tensordir(tensordir, verbose=False).filter(lambda f: vipy.util.filebase(vipy.util.noextension(f)) in trainsetid)
    trainloader = torch.utils.data.DataLoader(traintensor, num_workers=num_workers, batch_size=batchsize, pin_memory=True, shuffle=True)
    valtensor = vipy.torch.Tensordir(tensordir, verbose=False).filter(lambda f: vipy.util.filebase(vipy.util.noextension(f)) in valsetid)
    valloader = torch.utils.data.DataLoader(valtensor, num_workers=num_workers, batch_size=batchsize, pin_memory=True)
    t.fit(net, trainloader, valloader)


def train_noaug(trainsetid, valsetid, tensordir, outdir, batchsize=32, num_workers=4, resume_from_checkpoint=None, version=6):
    gc.disable()

    # Datasets: Replicate one video augmentation n times
    tensorset = vipy.torch.Tensordir(tensordir, verbose=False)
    d_videoid_to_dirlist = vipy.util.groupbyasdict(tensorset._dirlist, lambda f: vipy.util.filebase(f).split('_')[0])
    d_videoid_to_sample = {k:vipy.util.takeone(v) for (k,v) in d_videoid_to_dirlist.items()}
    d_videoid_to_auglist = {k:[v for i in range(len(d_videoid_to_dirlist[k]))] for (k,v) in d_videoid_to_sample.items()}
    tensorset._dirlist = vipy.util.flatlist(d_videoid_to_auglist.values())

    assert heyvi.version.is_at_least('0.2.26')  
    net = heyvi.recognition.CAP(labelset=version, modelfile=resume_from_checkpoint)  
    checkpoint_callback = ModelCheckpoint(save_top_k=-1, monitor='avg_val_loss', verbose=True, mode='min')  
    t = pl.Trainer(gpus=[0,1,2,3,4,5,6,7], accelerator='ddp', default_root_dir=outdir, resume_from_checkpoint=resume_from_checkpoint, plugins=DDPPlugin(find_unused_parameters=False), callbacks=[checkpoint_callback])

    traintensor = tensorset.clone().filter(lambda f: vipy.util.filebase(vipy.util.noextension(f)) in trainsetid)
    valtensor = tensorset.clone().filter(lambda f: vipy.util.filebase(vipy.util.noextension(f)) in valsetid)
    trainloader = torch.utils.data.DataLoader(traintensor, num_workers=num_workers, batch_size=batchsize, pin_memory=True, shuffle=True)
    valloader = torch.utils.data.DataLoader(valtensor, num_workers=num_workers, batch_size=batchsize, pin_memory=True)
    t.fit(net, trainloader, valloader)


def train_noaway(trainsetid, valsetid, tensordir, outdir, batchsize=32, num_workers=4, resume_from_checkpoint=None, version=6):
    gc.disable()

    # Datasets: Remove (facing away) from training and validation set
    (trainset, valset) = trainvalset('cap_classification_pad')
    iid = set([v.videoid() for v in trainset.union(valset) if '(facing away)' not in v.getattribute('collection_name')])  # to keep
    tensorset = vipy.torch.Tensordir(tensordir, verbose=False).filter(lambda f,iid=iid: vipy.util.filebase(f).split('_')[0] in iid)  # to keep

    assert heyvi.version.is_at_least('0.2.26')  
    net = heyvi.recognition.CAP(labelset=version, modelfile=resume_from_checkpoint)  
    checkpoint_callback = ModelCheckpoint(save_top_k=-1, monitor='avg_val_loss', verbose=True, mode='min')  
    t = pl.Trainer(gpus=[0,1,2,3,4,5,6,7], accelerator='ddp', default_root_dir=outdir, resume_from_checkpoint=resume_from_checkpoint, plugins=DDPPlugin(find_unused_parameters=False), callbacks=[checkpoint_callback])

    traintensor = tensorset.clone().filter(lambda f: vipy.util.filebase(vipy.util.noextension(f)) in trainsetid)
    valtensor = tensorset.clone().filter(lambda f: vipy.util.filebase(vipy.util.noextension(f)) in valsetid)
    trainloader = torch.utils.data.DataLoader(traintensor, num_workers=num_workers, batch_size=batchsize, pin_memory=True, shuffle=True)
    valloader = torch.utils.data.DataLoader(valtensor, num_workers=num_workers, batch_size=batchsize, pin_memory=True)
    t.fit(net, trainloader, valloader)


def train_supercollector(trainsetid, valsetid, tensordir, outdir, batchsize=32, num_workers=4, resume_from_checkpoint=None, version=6):
    gc.disable()

    # Datasets: Remove (facing away) from training and validation set
    (trainset, valset) = trainvalset('cap_classification_pad')
    d_collectorid_to_videos = {k:len(v) for (k,v) in vipy.util.groupbyasdict(trainset, lambda v: v.getattribute('collector_id')).items()}
    iid = set([v.videoid() for v in trainset if d_collectorid_to_videos[v.getattribute('collector_id')] > 3250])  # 499308/777902 = 65%
    traintensor = vipy.torch.Tensordir(tensordir, verbose=False).filter(lambda f,iid=iid: vipy.util.filebase(f).split('_')[0] in iid) # to keep
    valtensor = vipy.torch.Tensordir(tensordir, verbose=False)

    assert heyvi.version.is_at_least('0.2.26')  
    net = heyvi.recognition.CAP(labelset=version, modelfile=resume_from_checkpoint)  
    checkpoint_callback = ModelCheckpoint(save_top_k=-1, monitor='avg_val_loss', verbose=True, mode='min')  
    t = pl.Trainer(gpus=[0,1,2,3,4,5,6,7], accelerator='ddp', default_root_dir=outdir, resume_from_checkpoint=resume_from_checkpoint, plugins=DDPPlugin(find_unused_parameters=False), callbacks=[checkpoint_callback])

    traintensor = traintensor.filter(lambda f: vipy.util.filebase(vipy.util.noextension(f)) in trainsetid)
    valtensor = valtensor.filter(lambda f: vipy.util.filebase(vipy.util.noextension(f)) in valsetid)
    trainloader = torch.utils.data.DataLoader(traintensor, num_workers=num_workers, batch_size=batchsize, pin_memory=True, shuffle=True)
    valloader = torch.utils.data.DataLoader(valtensor, num_workers=num_workers, batch_size=batchsize, pin_memory=True)
    t.fit(net, trainloader, valloader)


def train_coarselabel(trainsetid, valsetid, tensordir, outdir, batchsize=32, num_workers=4, resume_from_checkpoint=None, version=6, do_trainingweights=False):
    gc.disable()

    d_label_to_coarselabel = vipy.util.load('cap_label_to_superlabel.json')
    if do_trainingweights:
        (trainset, valset) = trainvalset('cap_classification_pad')
        D = trainset.localmap(lambda v: v.category(d_label_to_coarselabel[v.category()]).activitymap(lambda a: a.category(d_label_to_coarselabel[a.category()])))
        print(vipy.util.writecsv([(k,v) for (k,v) in D.class_to_index().items()], os.path.join(outdir, 'cap_classification_pad_coarse_class_to_index.csv')))
        W = D.multilabel_inverse_frequency_weight()  # cap inverse frequency weights        
        print(vipy.util.writecsv(W.items(), os.path.join(outdir, 'cap_classification_pad_coarse_class_to_training_weight.csv')))
        return

    # Replace fully connected layer to match new class dimensionality
    assert resume_from_checkpoint is not None
    d = torch.load(resume_from_checkpoint)
    (w,b) = (d['state_dict']['net.fc.weight'],  d['state_dict']['net.fc.bias'])
    d_class_to_index = {k:int(v) for (k,v) in vipy.util.readcsv(os.path.join(outdir, 'cap_classification_pad_coarse_class_to_index.csv'))}
    if w.shape[0] != len(d_class_to_index):
        w = torch.randn_like(w)[0:len(d_class_to_index), :]
        b = torch.randn_like(b)[0:len(d_class_to_index)]
        d['state_dict']['net.fc.weight'] = w
        d['state_dict']['net.fc.bias'] = b
        d['optimizer_states'] = []        
        resume_from_checkpoint = vipy.util.mktemp('ckpt')
        torch.save(d, resume_from_checkpoint)


    assert heyvi.version.is_at_least('0.2.26')  
    net = heyvi.recognition.CAP(labelset='cap_coarse_v2', modelfile=resume_from_checkpoint)  
    checkpoint_callback = ModelCheckpoint(save_top_k=-1, monitor='avg_val_loss', verbose=True, mode='min')  
    t = pl.Trainer(gpus=[0,1,2,3,4,5,6,7], accelerator='ddp', default_root_dir=outdir, resume_from_checkpoint=resume_from_checkpoint, plugins=DDPPlugin(find_unused_parameters=False), callbacks=[checkpoint_callback])

    # Datasets: Mutate labels to be coarse class
    to_coarselabel = lambda Y,d=d_label_to_coarselabel: [[d[yi] for yi in y] for y in Y]
    tensorset = vipy.torch.Tensordir(tensordir, verbose=False, mutator=to_coarselabel)
    traintensor = tensorset.clone().filter(lambda f: vipy.util.filebase(vipy.util.noextension(f)) in trainsetid)
    valtensor = tensorset.clone().filter(lambda f: vipy.util.filebase(vipy.util.noextension(f)) in valsetid)

    trainloader = torch.utils.data.DataLoader(traintensor, num_workers=num_workers, batch_size=batchsize, pin_memory=True, shuffle=True)
    valloader = torch.utils.data.DataLoader(valtensor, num_workers=num_workers, batch_size=batchsize, pin_memory=True)
    t.fit(net, trainloader, valloader)


def ad_montage(outfile):
    D = [v for f in vipy.util.findjson(DATASET['cap_detection_handheld']['truthjson']) for v in vipy.util.load(f)]
    D = vipy.util.permutelist(D)[0:10]
    d = {v.getattribute('collection_name'):v for v in D}

    f = vipy.image.mutator_show_verb_only()
    imlist = [[f(v.frame(a.middleframe())).objectcrop(dilate=2.0, maxsquare=True).mindim(256).annotate().setattribute('collection_name', v.getattribute('collection_name')) for a in v.activitylist()][0:7] for v in d.values() if len(v.activitylist())>=7]
    print([im[0].getattribute('collection_name') for im in imlist])
    return vipy.visualize.montage(vipy.util.flatlist(imlist), 256,256,len(imlist),7).saveas(outfile)

def ad_handheld_videomontage(outfile):
    D = [v for f in vipy.util.findjson(DATASET['cap_detection_handheld']['truthjson']) for v in vipy.util.load(f)]
    D = vipy.util.permutelist(D)
    d = {v.getattribute('collection_name'):v for v in D}
    f = vipy.image.mutator_show_verb_only()

    collections = [k for k in list(d.keys())][0:28]  # choose first 28
    print(collections)

    videolist = [d[k].trackcrop(dilate=2.0, maxsquare=True).mindim(256).framerate(30.0).annotate(mutator=f) for k in collections]
    return vipy.visualize.videomontage(videolist, 256,256, gridrows=4, gridcols=7).saveas(outfile)

def ad_rigid_videomontage(outfile):
    D = [v for f in vipy.util.findjson(DATASET['cap_detection_rigid']['truthjson']) for v in vipy.util.load(f)]
    D = vipy.util.permutelist(D)
    d = {v.getattribute('collection_name'):v for v in D}
    f = vipy.image.mutator_show_verb_only()

    collections = [k for k in list(d.keys())][0:28]  # choose first 28
    print(collections)

    videolist = [d[k].clip(d[k].first_activity().startframe(), d[k].last_activity().endframe()).trackcrop(dilate=1.2, maxsquare=True).mindim(256).framerate(30.0).annotate(mutator=f) for k in collections]
    return vipy.visualize.videomontage(videolist, 256,256, gridrows=4, gridcols=7).saveas(outfile)


def ac_bylabel(outfile='ac_bylabel.pdf', stabilized=True, csvfile=None):
    import pycollector.program.pip
    d = pycollector.program.pip.dashboard()

    if csvfile is None:
        csvfile = './cap_benchmark/cap_classification_pad_stabilized/cap_classification_pad_stabilized/activity_scores.csv' if stabilized else './cap_benchmark/cap_classification_pad/cap_classification_pad/activity_scores.csv'
    csv = sorted([(c,float(s)) for (c,t,s) in vipy.util.readcsv(csvfile, ignoreheader=True)], key=lambda x: x[1], reverse=True)
    activities = set([x for (x,y) in csv])
    d_category_to_activitytype_color = {c:'red' if d.labels.is_person_person_activity(c) else ('green' if (d.labels.is_person_object_activity(c) or d.labels.is_object_person_activity(c)) else 'blue') for c in activities}
    
    colors = vipy.show.colorlist()                
    d_label_to_superlabel = vipy.util.readjson('cap_label_to_superlabel.json')  # same directory as script
    d_superlabel_to_color = {v:colors[k%len(colors)] for (k,v) in enumerate(set(d_label_to_superlabel.values()))}
    #print(vipy.metrics.histogram([y for (x,y) in csv], [x for (x,y) in csv], outfile=outfile, ylabel='Mean AP', fontsize=1, barcolors=[d_superlabel_to_color[d_label_to_superlabel[x]] if x in d_label_to_superlabel else 'black' for (x,y) in csv]))

    plt.figure(1)
    colors = {'Person Activity':'blue', 'Person/Object Activity' :'green', 'Person/Person Activity':'red'}         
    labels = list(colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
    plt.legend(handles, labels, loc='upper right')
    print(vipy.metrics.histogram([y for (x,y) in csv], [x for (x,y) in csv], figure=1, outfile=outfile, ylabel='Mean AP', fontsize=1, barcolors=[d_category_to_activitytype_color[x] if x in d_category_to_activitytype_color else 'black' for (x,y) in csv]))


def ac_activitytype(outfile='cap_category_to_activitytype.json'):
    import pycollector.program.pip
    d = pycollector.program.pip.dashboard()

    d_category_to_activitytype = {c:'person_person' if d.labels.is_person_person_activity(c) else ('person_object' if (d.labels.is_person_object_activity(c) or d.labels.is_object_person_activity(c) or c.startswith('car_')) else 'person') for c in d.labels.activities()}
    vipy.util.writejson(d_category_to_activitytype, outfile)
    

def confusion_graph(outcsv, refcsv, outfile, tau=0.07):
    import igraph 

    d_label_to_superlabel = vipy.util.readjson('cap_label_to_superlabel.json')  # same directory as script
    d_iid_to_category = {iid:c for (iid,c) in vipy.util.readcsv(refcsv, ignoreheader=True)}
    d_category_to_index = {c:k for (k,c) in enumerate(set([x[1] for x in vipy.util.readcsv(outcsv, ignoreheader=True)]).union([x[1] for x in vipy.util.readcsv(refcsv, ignoreheader=True)]))}   # FIXME: why empty?
    d_index_to_category = {v:k for (k,v) in d_category_to_index.items()}
    colorlist = list(igraph.drawing.colors.known_colors.keys())
    d_superlabel_to_color = {c:colorlist[k%len(colorlist)] for (k,c) in enumerate(set(d_label_to_superlabel.values()))}
    d_label_to_superlabel['person_highfives_hand'] = 'person_gestures'  # FIXME: WHY?

    confusion_matrix = np.zeros( (len(d_category_to_index), len(d_category_to_index)) )
    for (iid, c, score) in vipy.util.readcsv(outcsv, ignoreheader=True):
        confusion_matrix[d_category_to_index[d_iid_to_category[iid]], d_category_to_index[c]] += float(score) if len(score)>0 else 0
    cm = vipy.linalg.rowstochastic(confusion_matrix)

    confused = sorted([((i,j), cm[i,j]) for i in range(len(cm)) for j in range(len(cm)) if i != j], key=lambda x: float(x[1]), reverse=True)
    topk = [(d_index_to_category[i], d_index_to_category[j]) for ((i,j),v) in confused[0:50]]
    print(topk)

    # Generate graph visualization: vertex label=fine class, vertex color=coarse class, edge=confused, edge thickness=confusion weight
    g = igraph.Graph()    
    g.add_vertices(len(d_category_to_index))
    g.add_edges( [(int(i),int(j)) for ((i,j),w) in confused if float(w)>tau] )
    g.es['weight'] = [3*float(w) for ((i,j),w) in confused if float(w)>tau]
    visual_style = {}
    visual_style["vertex_frame_color"] = 'grey'    
    visual_style["edge_color"] = 'grey'    
    visual_style["vertex_size"] = 6
    visual_style["vertex_label_size"] = 3
    visual_style["edge_curved"] = False
    visual_style["edge_width"] = g.es['weight']
    g.vs["label"] = [d_index_to_category[i].replace('person_','') for i in range(len(d_category_to_index))]
    g.vs["color"] = [d_superlabel_to_color[d_label_to_superlabel[d_index_to_category[i]]] if (len(d_index_to_category[i])>0 and d_index_to_category[i] in d_label_to_superlabel) else 'grey' for i in range(len(d_category_to_index)) ]     
    layout = g.layout_fruchterman_reingold()
    igraph.plot(g, outfile, layout=layout, **visual_style)
    return outfile


def confusion_matrix(outcsv, refcsv, outfile):
    d_iid_to_category = {iid:c for (iid,c) in vipy.util.readcsv(refcsv, ignoreheader=True)}
    d_category_to_index = {c:k for (k,c) in enumerate(set([x[1] for x in vipy.util.readcsv(outcsv, ignoreheader=True)]).union([x[1] for x in vipy.util.readcsv(refcsv, ignoreheader=True)]))}
    d_index_to_category = {v:k for (k,v) in d_category_to_index.items()}
    confusion_matrix = np.zeros( (len(d_category_to_index), len(d_category_to_index)) )
    for (iid, c, score) in vipy.util.readcsv(outcsv, ignoreheader=True):
        confusion_matrix[d_category_to_index[d_iid_to_category[iid]], d_category_to_index[c]] += float(score) if len(score)>0 else 0
    cm = vipy.linalg.rowstochastic(confusion_matrix)
    return vipy.metrics.confusion_matrix(cm, outfile=outfile, classes=list(d_category_to_index.keys()), fontsize=1, figsize=(12,12))

def ac_topk(outcsv, refcsv, topk=1):
    d_iid_to_truth = {iid:c for (iid, c) in vipy.util.readcsv(refcsv, ignoreheader=True)}
    d_iid_to_sorted_pred = {k:set([z[1] for z in sorted(v, key=lambda y: float(y[2]) if len(y[2])>0 else 0, reverse=True)][0:topk]) for (k,v) in vipy.util.groupbyasdict(vipy.util.readcsv(outcsv, ignoreheader=True), lambda x: x[0]).items()}
    f = [(iid in d_iid_to_sorted_pred and c in d_iid_to_sorted_pred[iid]) for (iid, c) in d_iid_to_truth.items()]
    return float(sum(f)) / float(len(d_iid_to_truth))
            
    
def test(model, dataset, outdir, scoring=True, processing=True, printevery=100, confusion=False, take=None):    
    assert dataset in DATASET 
    assert model in MODEL
    assert scoring or processing

    outdir = vipy.util.remkdir(os.path.join(outdir, dataset, model))
    testref = DATASET[dataset]['testref']
    scorerref = os.path.join(outdir, '%s_ref.csv' % dataset)
    outcsv = os.path.join(outdir, '%s_classification_output.csv' % dataset)
    scorercsv = os.path.join(outdir, '%s_sys.csv' % dataset)
    truthiid = DATASET[dataset]['truthiid']

    S = heyvi.system.CAP(modelfile=MODEL[model]['modelpath'], labelset=MODEL[model]['labelset'], verbose=False)
    D = testset(dataset)
    if take is not None:
        D = D.take(int(take), seed=42)  

    gc.disable()
    if dataset.startswith('cap_classification'):
        if processing:
            idx2iid = {v:k for (k,v) in vipy.util.readjson(truthiid).items()}
            V = [vipy.util.trycatcher(lambda v,s=S,k=k,n=len(D),m=printevery: s.classify(v.clone(), topk=None).printif(k%m==0, prefix='[cap_benchmark.test_classification][%d/%d]: ' % (k, n)), v) for (k,v) in enumerate(D, start=1)]
            #V = [S.classify(v.clone(), topk=None).print() for (k,v) in enumerate(D, start=1)]
            V = [v for v in V if v is not None]  # remove errors (why?)
            print('[cap_benchmark.test_classification]: errors=%s' % (str(set([d.instanceid() for d in D]).difference(set([v.instanceid() for v in V])))))  # FIXME: why?

            iidset = set([x[0] for x in vipy.util.readcsv(testref, ignoreheader=True)]).intersection(set([idx2iid[v.instanceid()] for v in V]))
            vipy.util.writecsv(vipy.util.flatlist([[(idx2iid[v.instanceid()], '', '')] if v.primary_activity() is None else [(idx2iid[v.instanceid()], a.category(), a.confidence()) for a in v.activitylist()]
                                                   for v in V]), outcsv, header=('video_file_id', 'activity_id', 'confidence_score'))
        if scoring:
            iidset = set([x[0] for x in vipy.util.readcsv(testref, ignoreheader=True)]).intersection(set([x[0] for x in vipy.util.readcsv(outcsv, ignoreheader=True)]))
            vipy.util.writecsv(vipy.util.dedupe([(iid, aid) for (iid, aid) in vipy.util.readcsv(testref, ignoreheader=True) if iid in iidset], lambda x: x[0]), scorerref, header=('video_file_id', 'activity_id'), comment='#')
            vipy.util.writecsv([(iid, aid, c) for (iid, aid, c) in vipy.util.readcsv(outcsv, ignoreheader=True) if iid in iidset], scorercsv, header=('video_file_id', 'activity_id', 'confidence_score'), comment='#')

            print('[cap_benchmark.test]: ac top-1=%f' % ac_topk(scorercsv, scorerref, topk=1))
            print('[cap_benchmark.test]: ac top-5=%f' % ac_topk(scorercsv, scorerref, topk=5))

            cmd = 'fad-scorer --verbose score-ac --filter_top_n=0  -r %s -y %s -o %s' % (scorerref, scorercsv, outdir)
            print('[cap_benchmark.test]: executing "%s"' % cmd); os.system(cmd)

            if confusion:
                print(confusion_matrix(outcsv=scorercsv, refcsv=scorerref, outfile=os.path.join(outdir, '%s_confusion_matrix.pdf' % dataset)))
                print(confusion_graph(outcsv=scorercsv, refcsv=scorerref, outfile=os.path.join(outdir, '%s_confusion_graph.pdf' % dataset), tau=0.07))

    elif dataset.startswith('cap_detection'):
        if processing:
            V = [vipy.util.trycatcher(lambda v,s=S,k=k,n=len(D),m=printevery: s.detect(v.clone()).printif(k%m==0, prefix='[cap_benchmark.test_detection][%d/%d]: ' % (k, n)), v) for (k,v) in enumerate(D, start=1)]
            V = [v for v in V if v is not None]  # remove errors (why?)
            print('[cap_benchmark.test_detection]: errors=%s' % (str(set([d.instanceid() for d in D]).difference(set([v.instanceid() for v in V])))))  # why?
            iidset = set([x[0] for x in vipy.util.readcsv(testref, ignoreheader=True)]).intersection(set([v.videoid() for v in V]))
            header = ('video_file_id', 'activity_id', 'confidence_score', 'frame_start', 'frame_end')
            vipy.util.writecsv(vipy.util.flatlist([[(v.videoid(), a.category(), a.confidence(), a.startframe(), a.endframe()) for a in v.activitylist()] if len(v.activitylist())>0 else [(v.videoid(),'','','','')] for v in V]), outcsv, header=header)
        if scoring:
            iidset = set([x[0] for x in vipy.util.readcsv(testref, ignoreheader=True)]).intersection(set([x[0] for x in vipy.util.readcsv(outcsv, ignoreheader=True)]))
            vipy.util.writecsv(list(set([tuple(r) for r in vipy.util.readcsv(testref, ignoreheader=True) if r[0] in iidset])), scorerref, header=('video_file_id', 'activity_id', 'frame_start', 'frame_end'), comment='#')
            vipy.util.writecsv(list(set([tuple(r) for r in vipy.util.readcsv(outcsv, ignoreheader=True) if r[0] in iidset])), scorercsv, header=('video_file_id', 'activity_id', 'confidence_score', 'frame_start', 'frame_end'), comment='#')
            cmd = 'fad-scorer --verbose score-tad -j 20 -i 0.2,0.5,0.8 -r %s -y %s -o %s' % (scorerref, scorercsv, outdir)
            print('[cap_benchmark.test]: executing "%s"' % cmd); os.system(cmd)            

            # FIXME: why are there duplicates in testref?

    else:
        raise ValueError('unknown task')
    gc.enable()
    

def mean_clips_per_class():
    D = vipy.util.load('../pip/final_release/refine.pkl')
    A = [a.category() for v in D for a in v.activitylist()]
    print(np.mean([len(v) for (k,v) in vipy.util.groupbyasdict(A, lambda a: a).items()]))  # 2835
    print(np.mean([len(v) for (k,v) in vipy.util.groupbyasdict(A, lambda a: a).items() if len(v)>1000])) # 3293.032710280374
    print(np.mean([len(v) for (k,v) in vipy.util.groupbyasdict(A, lambda a: a).items() if len(v)>2000])) # 4203
    np.mean([len(v) for (k,v) in vipy.util.groupbyasdict(A, lambda a: a).items() if len(v)>2300])  # top-250, 4501
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Dataset", required=True)
    parser.add_argument("--outdir", help="Output directory for generated files", default='./cap_benchmark')
    parser.add_argument("--tensorset", help="Generate compressed tensors for dataset", action='store_true')
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--noaug", help="Train a model without video augmentation", action='store_true')
    parser.add_argument("--noaway", help="Train a model without bias engineered collections", action='store_true')
    parser.add_argument("--supercollector", help="Train a model using only top-100 collectors", action='store_true')        
    parser.add_argument("--coarselabel", help="Train a model using coarse labelset", action='store_true')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--model", help="Model name for evaluation")
    parser.add_argument("--workers", help="Number of parallel workers", default=4)
    parser.add_argument("--resume", help="path to .ckpt to resume from")
    parser.add_argument("--confusion", help="Output confusion matrix and confusion graph (should only be used for fine grained datasets)", action='store_true')
    parser.add_argument("--score", help="Score only using pregenerated processing csv files, do not process videos", action='store_true')
    parser.add_argument("--take", help="Number of instances to take from dataset for evaluation")
    args = parser.parse_args()

    assert args.dataset in DATASET        
    assert not args.test or args.model is not None
    
    outdir = os.path.abspath(args.outdir)
    take = int(args.take) if (args.take is not None and args.take.lower() != 'none') else None
    if args.tensorset:
        (trainset, valset) = trainvalset(args.dataset)        
        tensorset(trainset.union(valset), outdir=outdir)
    if args.train and args.coarselabel:        
        (trainsetid, valsetid) = (set(vipy.util.load(DATASET[args.dataset]['trainsplit'])['train']), set(vipy.util.load(DATASET[args.dataset]['trainsplit'])['val']))
        (version, resume_from_checkpoint) = ('cap', './cap_benchmark/cap_classification_pad_e18s57740.ckpt') if 'stabilize' not in args.dataset else ('cap_stabilized', './cap_benchmark/cap_classification_pad_stabilized_e18s56790v8.ckpt')
        train_coarselabel(trainsetid, valsetid, tensordir=os.path.join(outdir, args.dataset, 'tensorset'), outdir=vipy.util.remkdir(os.path.join(outdir, args.dataset, 'coarselabel')), num_workers=int(args.workers), version=version, resume_from_checkpoint=resume_from_checkpoint, do_trainingweights=False)        
    elif args.train and args.noaway:        
        (trainsetid, valsetid) = (set(vipy.util.load(DATASET[args.dataset]['trainsplit'])['train']), set(vipy.util.load(DATASET[args.dataset]['trainsplit'])['val']))
        (version, resume_from_checkpoint) = ('cap', './cap_benchmark/cap_classification_pad_e18s57740.ckpt') if 'stabilize' not in args.dataset else ('cap_stabilized', './cap_benchmark/cap_classification_pad_stabilized_e18s56790v8.ckpt')
        train_noaway(trainsetid, valsetid, tensordir=os.path.join(outdir, args.dataset, 'tensorset'), outdir=vipy.util.remkdir(os.path.join(outdir, args.dataset, 'noaway')), num_workers=int(args.workers), version=version, resume_from_checkpoint=resume_from_checkpoint)
    elif args.train and args.supercollector:        
        (trainsetid, valsetid) = (set(vipy.util.load(DATASET[args.dataset]['trainsplit'])['train']), set(vipy.util.load(DATASET[args.dataset]['trainsplit'])['val']))
        (version, resume_from_checkpoint) = ('cap', './cap_benchmark/cap_classification_pad_e18s57740.ckpt') if 'stabilize' not in args.dataset else ('cap_stabilized', './cap_benchmark/cap_classification_pad_stabilized_e18s56790v8.ckpt')
        train_supercollector(trainsetid, valsetid, tensordir=os.path.join(outdir, args.dataset, 'tensorset'), outdir=vipy.util.remkdir(os.path.join(outdir, args.dataset, 'supercollector')), num_workers=int(args.workers), version=version, resume_from_checkpoint=resume_from_checkpoint)                
    elif args.train and args.noaug:
        (trainsetid, valsetid) = (set(vipy.util.load(DATASET[args.dataset]['trainsplit'])['train']), set(vipy.util.load(DATASET[args.dataset]['trainsplit'])['val']))
        (version, resume_from_checkpoint) = ('cap', './cap_benchmark/cap_classification_pad_e18s57740.ckpt') if 'stabilize' not in args.dataset else ('cap_stabilized', './cap_benchmark/cap_classification_pad_stabilized_e18s56790v8.ckpt')
        train_noaug(trainsetid, valsetid, tensordir=os.path.join(outdir, args.dataset, 'tensorset'), outdir=vipy.util.remkdir(os.path.join(outdir, args.dataset, 'noaug')), num_workers=int(args.workers), version=version, resume_from_checkpoint=resume_from_checkpoint)
    elif args.train and not args.noaug:
        (trainsetid, valsetid) = (set(vipy.util.load(DATASET[args.dataset]['trainsplit'])['train']), set(vipy.util.load(DATASET[args.dataset]['trainsplit'])['val']))
        train(trainsetid, valsetid, tensordir=os.path.join(outdir, args.dataset, 'tensorset'), outdir=vipy.util.remkdir(os.path.join(outdir, args.dataset)), num_workers=int(args.workers), version = 7 if 'stabilize' in args.dataset else 6, resume_from_checkpoint=args.resume)
    if args.test:
        test(args.model, args.dataset, args.outdir, confusion=args.confusion, scoring=True, processing=not args.score, take=take)
