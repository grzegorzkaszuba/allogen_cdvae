



class StructureOptimizer:
    def __init__(self, cfg, path_out, loaders, num_steps=10, elements=('Cr', 'Fe', 'Ni')):
        self.cfg = cfg
        self.best_by_composition = {}
        self.path_out = path_out
        self.writer = SummaryWriter(path_out)
        self.num_steps = num_steps
        self.exp_cfg = initialize_exp_cfg(elements)

        self.n_structures_retrain = 0
        self.current_step = 0

        self.cur_train_loader = copy.deepcopy(loaders[0])
        self.cur_val_loader = copy.deepcopy(loaders[1])
        self.cur_structure_loader = copy.deepcopy(loaders[2])
        os.makedirs(path_out, exist_ok=True)


    def step(self):
        # Cycle Setup
        stepdir = os.path.join(self.path_out, str(self.current_step))
        cif_dir = os.path.join(stepdir, 'cif_raw')
        lammpsdata_dir = os.path.join(stepdir, 'lammps_raw')
        relaxed_cif_dir = os.path.join(stepdir, 'cif_relaxed')
        relaxed_lammpsdata_dir = os.path.join(stepdir, 'lammps_relaxed')
        chkdir = os.path.join(stepdir, 'checkpoints')
        pt_dir = os.path.join(stepdir, 'pt')
        for directory in [stepdir, cif_dir, lammpsdata_dir, relaxed_cif_dir, relaxed_lammpsdata_dir, chkdir, pt_dir]:
            os.makedirs(directory, exist_ok=True)

    def create_trainer(self, directory='initial_training'):
        trainer = get_trainer(cfg, os.path.join(self.path_out, directory))


