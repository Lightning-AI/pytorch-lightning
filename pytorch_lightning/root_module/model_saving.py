class ModelIO(object):

    def on_load_checkpoint(self, checkpoint):
        """
        Do something with the checkpoint
        Gives model a chance to load something before state_dict is restored
        :param checkpoint:
        :return:
        """
        pass

    def on_save_checkpoint(self, checkpoint):
        """
        Give the model a chance to add something to the checkpoint.
        state_dict is already there
        """
        pass

    # -------------------------
    # OPTIONAL HOOKS
    # -------------------------
    def on_hpc_save(self, checkpoint):
        """
        Hook to do whatever you need right before Slurm manager saves the model
        :return:
        """
        pass

    def on_hpc_load(self, checkpoint):
        """
        Hook to do whatever you need right before Slurm manager loads the model
        :return:
        """
        pass
