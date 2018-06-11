from Utilities import *
from Models import *

###############################################################################
def time_seed ():
    seed = None
    while seed == None:
        cur_time = time.time ()
        seed = int ((cur_time - int (cur_time)) * 1000000)
    return seed

class ImageDataFlow(RNGDataFlow):
    def __init__(self, 
        imageDir, 
        labelDir, 
        size, 
        dtype='float32', 
        isTrain=False, 
        isValid=False, 
        isTest=False, 
        pruneLabel=False, 
        shape=[3, 320, 320]):

        self.dtype      = dtype
        self.imageDir   = imageDir
        self.labelDir   = labelDir
        self._size      = size
        self.isTrain    = isTrain
        self.isValid    = isValid

        imageFiles = natsorted (glob.glob(self.imageDir + '/*.*'))
        labelFiles = natsorted (glob.glob(self.labelDir + '/*.*'))
        print(imageFiles)
        print(labelFiles)
        self.images = []
        self.labels = []
        self.data_seed = time_seed ()
        self.data_rand = np.random.RandomState(self.data_seed)
        self.rng = np.random.RandomState(999)
        for imageFile in imageFiles:
            image = skimage.io.imread (imageFile)
            self.images.append(image)
        for labelFile in labelFiles:
            label = skimage.io.imread (labelFile)
            self.labels.append(label)
            
        self.DIMZ = shape[0]
        self.DIMY = shape[1]
        self.DIMX = shape[2]
        self.pruneLabel = pruneLabel

    def size(self):
        return self._size

    def AugmentPair(self, src_image, src_label, pipeline, seed=None, verbose=False):
        np.random.seed(seed) if seed else np.random.seed(2015)
        # print(src_image.shape, src_label.shape, aug_image.shape, aug_label.shape) if verbose else ''
        if src_image.ndim==2:
            src_image = np.expand_dims(src_image, 0)
            src_label = np.expand_dims(src_label, 0)
        
        # Create the result
        aug_images = [] #np.zeros_like(src_image)
        aug_labels = [] #np.zeros_like(src_label)
        
        # print(src_image.shape, src_label.shape)
        for z in range(src_image.shape[0]):
            #Image and numpy has different matrix order
            pipeline.set_seed(seed)
            aug_image = pipeline._execute_with_array(src_image[z,...]) 
            pipeline.set_seed(seed)
            aug_label = pipeline._execute_with_array(src_label[z,...])        
            aug_images.append(aug_image)
            aug_labels.append(aug_label)
        aug_images = np.array(aug_images).astype(np.float32)
        aug_labels = np.array(aug_labels).astype(np.float32)
        # print(aug_images.shape, aug_labels.shape)
        return aug_images, aug_labels
    ###############################################################################
    def random_reverse(self, image, seed=None):
        assert ((image.ndim == 2) | (image.ndim == 3))
        if seed:
            self.rng.seed(seed)
        random_reverse = self.rng.randint(1,3)
        if random_reverse==1:
            reverse = image[::1,...]
        elif random_reverse==2:
            reverse = image[::-1,...]
        image = reverse
        return image
    ###############################################################################
    def grow_boundaries(self, gt, steps=1, background=0):
        from scipy import ndimage
        foreground = np.zeros(shape=gt.shape, dtype=np.bool)
        masked = None
        
        for label in np.unique(gt):
            if label == background:
                continue
            label_mask = gt==label
            # Assume that masked out values are the same as the label we are
            # eroding in this iteration. This ensures that at the boundary to
            # a masked region the value blob is not shrinking.
            if masked is not None:
                label_mask = np.logical_or(label_mask, masked)
            eroded_label_mask = ndimage.binary_erosion(label_mask, iterations=steps, 
                                                       border_value=1)
            foreground = np.logical_or(eroded_label_mask, foreground)

        # label new background
        background = np.logical_not(foreground)
        gt[background] = 0
        
        return gt
    ###############################################################################
    def get_data(self):
        for k in range(self._size):
            #
            # Pick randomly a tuple of training instance
            #
            rand_index = self.data_rand.randint(0, len(self.images))
            image_p = self.images[rand_index].copy ()
            label_p = self.labels[rand_index].copy ()

            seed = time_seed () #self.rng.randint(0, 20152015)
            
                        # Downsample here
            #pz = self.data_rand.randint(0, 2)
            py = self.data_rand.randint(0, 3)
            px = self.data_rand.randint(0, 3)
            image_p = image_p[::1, py::3, px::3]
            label_p = label_p[::1, py::3, px::3]
            # Cut 1 or 3 slices along z, by define DIMZ, the same for paired, randomly for unpaired

            dimz, dimy, dimx = image_p.shape
            # The same for pair
            randz = self.data_rand.randint(0, dimz-self.DIMZ+1)
            randy = self.data_rand.randint(0, dimy-self.DIMY+1)
            randx = self.data_rand.randint(0, dimx-self.DIMX+1)

            image_p = image_p[randz:randz+self.DIMZ,randy:randy+self.DIMY,randx:randx+self.DIMX]
            label_p = label_p[randz:randz+self.DIMZ,randy:randy+self.DIMY,randx:randx+self.DIMX]
            
            if self.isTrain:
                # Augment the pair image for same seed
                p_train = Augmentor.Pipeline()
                p_train.rotate_random_90(probability=0.75, resample_filter=Image.NEAREST)
                p_train.rotate(probability=1, max_left_rotation=20, max_right_rotation=20, resample_filter=Image.NEAREST)
                # p_train.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=5)
                p_train.flip_random(probability=0.75)

                image_p, label_p = self.AugmentPair(image_p.copy(), label_p.copy(), p_train, seed=seed)
                
                image_p = self.random_reverse(image_p, seed=seed)
                label_p = self.random_reverse(label_p, seed=seed)
                


            # # Calculate linear label
            if self.pruneLabel:
                label_p, nb_labels_p = skimage.measure.label(label_p.copy(), return_num=True)        

            # if self.grow_boundaries
            label_p = self.grow_boundaries(label_p)
            label_p[0,0,0] = 0 # hack for entire label is 1 due to CB3
            # Expand dim to make single channel
            image_p = np.expand_dims(image_p, axis=-1)
            label_p = np.expand_dims(label_p, axis=-1)

            #Return the membrane
            affnt_p = np_seg_to_aff(label_p)
           
            yield [image_p.astype(np.float32), 
                   affnt_p.astype(np.float32), 
                   label_p.astype(np.float32), 
                   ] 

###############################################################################
def get_data(dataDir, isTrain=False, isValid=False, isTest=False, shape=[16, 320, 320]):
    # Process the directories 
    if isTrain:
        num=500
        names = ['trainA', 'trainB']
    if isValid:
        num=1
        names = ['trainA', 'trainB']
    if isTest:
        num=10
        names = ['validA', 'validB']

    
    dset  = ImageDataFlow(os.path.join(dataDir, names[0]),
                               os.path.join(dataDir, names[1]),
                               num, 
                               isTrain=isTrain, 
                               isValid=isValid, 
                               isTest =isTest, 
                               shape=shape, 
                               pruneLabel=True)
    dset.reset_state()
    return dset
###############################################################################
class Model(ModelDesc):
    @auto_reuse_variable_scope
    def generator(self, img, last_dim=1, nl=INLReLU3D, nb_filters=32):
        assert img is not None
        img = tf.expand_dims(img, axis=0)
        ret = arch_fusionnet_translator_3d(img, last_dim=last_dim, nl=nl, nb_filters=nb_filters)
        ret = tf.squeeze(ret, axis=0)
        return ret 

    def inputs(self):
        return [
            tf.placeholder(tf.float32, (args.DIMZ, args.DIMY, args.DIMX, 1), 'image'),
            tf.placeholder(tf.float32, (args.DIMZ, args.DIMY, args.DIMX, 3), 'affnt'),
            tf.placeholder(tf.float32, (args.DIMZ, args.DIMY, args.DIMX, 1), 'label'),
            ]

    def build_graph(self, image, affnt, label):
        G = tf.get_default_graph()
        pi, pa, pl = image, affnt, label

        # Construct the graph
        with tf.variable_scope('gen'):
            with tf.device('/device:GPU:0'):
                with tf.variable_scope('image2affnt'):
                    pia = self.generator(tf_2tanh(pi), last_dim=3, nl=tf.nn.tanh, nb_filters=32)
        pia = tf_2imag(pia, maxVal=1.0)
        pia = tf.identity(pia, name='pia')
        # Define loss hre
        losses = [] 

        with tf.name_scope('loss_dice'):
            dice_ia = tf.identity(1.0 - dice_coe(pia, pa, axis=[0,1,2,3], loss_type='jaccard'), 
                                 name='dice_ia')  
            losses.append(1e2*dice_ia)
            add_moving_summary(dice_ia)

        with tf.name_scope('loss_wbce'):
            #wbce_ia = tf.identity(1.0 - wbce_coe(pia, pa, axis=[0,1,2,3], loss_type='jaccard'), 
            #                     name='wbce_ia')  
            # Reshape affinity
            input_shape = (args.DIMZ, args.DIMY, args.DIMX)
            nhood = malis.mknhood3d (1)
            affs_shape = (len(nhood),) + input_shape
            gt_affs = tf.transpose (tf.squeeze (affnt), perm=[3,0,1,2], name='gt_affs')
            gt_seg  = tf.squeeze (label) 
            logits = tf.transpose (tf.squeeze (pia), perm=[3,0,1,2], name='logits')
            affs = tf.identity (logits, name='affs')  
            ###
            pos_cnt = tf.cast (tf.count_nonzero (tf.cast (gt_affs, tf.int32)), tf.float32)
            neg_cnt = tf.cast (tf.constant (np.prod (affs_shape)), tf.float32) - pos_cnt
            pos_weight = neg_cnt / (pos_cnt+1e-9)
            summary.add_tensor_summary (pos_weight, types=['scalar'])
            weighted_bce_losses = tf.nn.weighted_cross_entropy_with_logits (targets=gt_affs, logits=logits, pos_weight=pos_weight)

            malis_weights, pos_weights, neg_weights = malis_weights_op(affs, gt_affs, gt_seg, nhood, name='malis_weights', limit_z=False)
            pos_weights = tf.identity (pos_weights, name='pos_weight')
            neg_weights = tf.identity (neg_weights, name='neg_weight')
            
            malis_weighted_bce_loss = tf.reduce_mean (tf.multiply (malis_weights, weighted_bce_losses), name='malis_weighted_bce_loss')

            losses.append(1e2*malis_weighted_bce_loss)
            add_moving_summary(malis_weighted_bce_loss)

        with tf.name_scope('loss_mae'):
            mae_ia = tf.reduce_mean(tf.abs(pa - pia), name='mae_ia')
            losses.append(1e0*mae_ia)
            add_moving_summary(mae_ia)

        # Aggregate final loss
        self.cost = tf.reduce_sum(losses, name='self.cost')
        add_moving_summary(self.cost)

        # Segmentation
        pz = tf.zeros_like(pi)
        viz = tf.concat([tf.concat([pi, 255*pa [...,0:1], 255*pa [...,1:2], 255*pa [...,2:3]], axis=2),
                         tf.concat([pl*20, 255*pia[...,0:1], 255*pia[...,1:2], 255*pia[...,2:3]], axis=2),
                         ], axis=1)
        viz = tf.cast(tf.clip_by_value(viz, 0, 255), tf.uint8, name='viz')
        tf.summary.image('labelized', viz, max_outputs=50)


    def optimizer(self):
        lr = symbolic_functions.get_scalar_var('learning_rate', 2e-4, summary=True)
        return tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-3)

###############################################################################
class VisualizeRunner(Callback):
    def __init__(self, input, tower_name='InferenceTower', device=0):
        self.dset = input 
        self._tower_name = tower_name
        self._device = device

    def _setup_graph(self):
        self.pred = self.trainer.get_predictor(
            ['image', 'affnt', 'label'], ['viz'])

    def _before_train(self):
        pass

    def _trigger(self):
        for lst in self.dset.get_data():
            viz_test = self.pred(lst)
            viz_test = np.squeeze(np.array(viz_test))
            self.trainer.monitors.put_image('viz_test', viz_test)

###############################################################################
def sample(dataDir, model_path, prefix='.'):

    
    print("Starting...")
    print(dataDir)
    imageFiles = glob.glob(os.path.join(dataDir, '*.tif'))
    print(imageFiles)
    # Load the model 
    predict_func = OfflinePredictor(PredictConfig(
        model=Model(),
        session_init=get_model_loader(model_path),
        input_names=['image'],
        output_names=['pia']))

    for imageFile in imageFiles:
        head, tail = os.path.split(imageFile)
        print tail
        affntFile = prefix+tail
        print affntFile

        # Read the image file
        image = skimage.io.imread(imageFile)

        #image = np.expand_dims(image, axis=-1)
        # convert to 3 channel image
        image = np.stack((image, image, image), -1)
        print(image.shape)
        def weighted_map_blocks(arr, inner, ghost, func=None): # work for 3D, inner=[1, 3, 3], ghost=[0, 2, 2], 
            dtype = np.float32 #arr.dtype

            arr = arr.astype(np.float32)
            # param
            outer = inner + 2*ghost
            outer = [(i + 2*g) for i, g in zip(inner, ghost)]
            shape = outer
            steps = inner
                
            print(outer)
            print(shape)
            print(inner)
            
            # pad the array
            padding = np.pad(arr, [[ghost[0], ghost[0]], 
                                   [ghost[1], ghost[1]], 
                                   [ghost[2], ghost[2]],  
                                   [ghost[3], ghost[3]]] , mode='symmetric') # mode='symmetric') #
            print(padding.shape)
            #print(padding)
            
            weights = np.zeros_like(padding)
            results = np.zeros_like(padding)
            
            v_padding = sliding_window_view(padding, shape, steps)
            v_weights = sliding_window_view(weights, shape, steps)
            v_results = sliding_window_view(results, shape, steps)
            
            def invert(val):
                #return 255-val 
                return val

            for z in range(v_padding.shape[0]):
                for y in range(v_padding.shape[1]):
                    for x in range(v_padding.shape[2]):
                        #for c in range(v_padding.shape[3]):
                        # Get the result
                        #v_result = invert(v_padding[z,y,x]) ### Todo function is here
                        v_result = np.array(func(
                                                (v_padding[z,y,x,0][...,0:1]) ) ) ### Todo function is here
                        v_results[z,y,x] += np.squeeze(v_result, axis=0)
                            
                        v_weight = np.ones_like(v_result)
                        v_weights[z,y,x] += v_weight
                            
            # Divided by the weight param
            results /= weights 
            
            
            current_shape = results.shape
            trimmed_shape = [np.arange(ghost[0]),(results.shape[0] - ghost[0]), 
                             np.arange(ghost[1]),(results.shape[1] - ghost[1]), 
                             np.arange(ghost[2]),(results.shape[2] - ghost[2]), 
                             np.arange(ghost[3]),(results.shape[3] - ghost[3]), 
                             ]
            # Trim the result
            results = results[(ghost[0]):(results.shape[0] - ghost[0]), 
                              (ghost[1]):(results.shape[1] - ghost[1]), 
                              (ghost[2]):(results.shape[2] - ghost[2]),
                              (ghost[3]):(results.shape[3] - ghost[3]),
                              ...]
            #results = results[trimmed_shape]
            
            return results.astype(dtype)
    

        #affnt = weighted_map_blocks(image, [20, 128, 128, 3], [2, 72, 72, 0], predict_func)
        affnt = weighted_map_blocks(image, [20, 256, 256, 3], [2, 8, 8, 0], predict_func)
        affnt = np.squeeze(affnt)
        skimage.io.imsave(affntFile, affnt)
    return None
###############################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',        default='0', help='comma seperated list of GPU(s) to use.')
    parser.add_argument('--data',  default='data/Kasthuri15/3D/', required=True, 
                                    help='Data directory, contain trainA/trainB/validA/validB')
    parser.add_argument('--load',   help='Load the model path')
    parser.add_argument('--DIMX',  type=int, default=320)
    parser.add_argument('--DIMY',  type=int, default=320)
    parser.add_argument('--DIMZ',  type=int, default=24)
    parser.add_argument('--sample', help='Run the deployment on an instance',
                                    action='store_true')
    global args
    args = parser.parse_args()
    
    # python Exp_FusionNet2D_-VectorField.py --gpu='0' --data='arranged/'

    
    train_ds = get_data(args.data, isTrain=True, isValid=False, isTest=False, shape=[args.DIMZ, args.DIMY, args.DIMX])
    valid_ds = get_data(args.data, isTrain=False, isValid=True, isTest=False, shape=[args.DIMZ, args.DIMY, args.DIMX])
    # test_ds  = get_data(args.data, isTrain=False, isValid=False, isTest=True)


    train_ds  = PrefetchDataZMQ(train_ds, 4)
    train_ds  = PrintData(train_ds)
    # train_ds  = QueueInput(train_ds)
    model     = Model()

    os.environ['PYTHONWARNINGS'] = 'ignore'

    # Set the GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Running train or deploy
    if args.sample:
        # TODO
        print("Deploy the data")
        sample(args.data, args.load, prefix='deploy_')
        # pass
    else:
        # Set up configuration
        # Set the logger directory
        logger.auto_set_dir()

        # Set up configuration
        config = TrainConfig(
            model           =   model, 
            dataflow        =   train_ds,
            callbacks       =   [
                PeriodicTrigger(ModelSaver(), every_k_epochs=50),
                PeriodicTrigger(VisualizeRunner(valid_ds), every_k_epochs=5),
                ScheduledHyperParamSetter('learning_rate', [(0, 2e-4), (100, 1e-4), (200, 2e-5), (300, 1e-5), (400, 2e-6), (500, 1e-6)], interp='linear'),
                ],
            max_epoch       =   500, 
            session_init    =   SaverRestore(args.load) if args.load else None,
            )
    
        # Train the model
        launch_train_with_config(config, QueueInputTrainer())






