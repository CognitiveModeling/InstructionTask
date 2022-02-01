import torch


class ActionInference():
    """Action inference class.

    An instance of this class provides the functionality to infer actions.

    Parameters
    ----------
    model : torch.nn.Module
        Recurrent neural network model which might be pretrained.
    policy : torch.Tensor
        Initial policy consisting of actions.
    optimizer : torch.optim.Optimizer
        Optimizer to optimize the policy with.
    inference_cycles : int
        Number of inference cycles.
    criterion : function
        Criterion for comparison of a list of predictions and a target.
    reset_optimizer : bool
        If True the optimizer's statistics are reset before each inference.
        If False the optimizer's statistics are kept across inferences.
    policy_handler : function
        Function that is applied to the policy after each optimization,
        e.g. can be used to keep policy in certain range.

    """

    def __init__(
            self, model, policy1, policy2, optimizer1, optimizer2, criterion, inference_cycles=50, num_blocks=3,
            reset_optimizer=True, policy_handler=lambda x: x, attention=False):

        assert (len(policy1.shape) == 3), "policy should be of shape (seq_len, batch, input_size)"
        assert (len(policy2.shape) == 3), "policy should be of shape (seq_len, batch, input_size)"
        assert (policy1.size(1) == 1), "batch of policy should be 1"
        assert (policy2.size(1) == 1), "batch of policy should be 1"

        self._model = model
        self.num_blocks = num_blocks
        self._policy1 = policy1
        self._policy2 = policy2
        self._policy1.requires_grad = True
        self._policy2.requires_grad = True
        self._inference_cycles = inference_cycles
        self._criterion = criterion
        self._optimizer1 = optimizer1
        self._optimizer2 = optimizer2
        self._reset_optimizer = reset_optimizer
        if self._reset_optimizer:
            self._optimizer1_orig_state = optimizer1.state_dict()
            self._optimizer2_orig_state = optimizer2.state_dict()
        self._policy_handler = policy_handler
        self.attention = attention
        n_agents = 3
        n_size = 1
        n_color = 3
        n_type = 1
        n_status = 1
        n_positions = 3
        n_positions_state = 6
        n_orientations = 6
        n_distances = 2
        # n_single_state = n_positions + n_orientations + n_distances + n_size + n_type + n_color + n_status
        n_single_state = n_color + n_size + n_positions_state + n_orientations + n_status
        self.block_sz = n_agents * n_single_state

    def reset_policies(self, p3, p5):
        pol3 = p3.clone()
        pol5 = p5.clone()

        for i in range(len(pol3[0][0])):
            pol3[0, 0, i] = 0.0
        for j in range(len(pol5[0][0])):
            pol5[0, 0, j] = 0.0
        return pol3, pol5

    def clamp(self, p1, p2):
        pol1 = p1.clone()
        pol2 = p2.clone()

        pol1 = torch.clamp(pol1, min=-1., max=1.)
        pol2 = torch.clamp(pol2, min=-1., max=1.)

        return pol1, pol2

    def predict(self, state, block, actionnum, orientationnum, first_block, current_block=None, testing=False):

        # Forward pass over policy
        # position = self._policy2.view(1, 13)
        if first_block:
            self._policy1 = torch.zeros([1, 1, 2]).to(device="cuda")
        position = actionnum
        position = torch.cat((position, self._policy1), dim=-1)
        position = torch.cat((position, orientationnum), dim=-1)
        position = torch.cat((position, self._policy2), dim=-1).view(1, 12)
        x = self._model(state, block, position, ai=self.attention, first_block=first_block, testing=testing)
        return x

    def action_inference(self, x, target, block, orientationnum, actionnum, current_blocks_in_game, first_block,
                         current_block=None, testing=False):
        """Optimize the current policy.

        Given an initial input, an initial hidden state, a context, and a
        target, this method infers a policy based on an imagination into the
        future.

        Parameters
        ----------
        x : torch.Tensor
            Initial input.
        state : torch.Tensor or tuple
            Initial hidden (and cell) state of the network.
        context : torch.Tensor
            Context activations over the whole prediction.
        target
            Target to compare prediction to.

        Returns
        -------
        policy : torch.Tensor
            Optimized policy.
        outputs : list
            Predicted observations corresponding to the optimized policy.
        states : list of torch.Tensor or list of tuple
            Hidden states of the model corresponding to the outputs.

        """

        assert (len(x.shape) == 3), "x should be of shape (seq_len, batch, input_size)"
        assert (x.size(0) == 1), "seq_len of x should be 1"
        assert (x.size(1) == 1), "batch of x should be 1"

        assert (len(target.shape) == 3), "target should be of shape (seq_len, batch, output_size)"
        # assert (target.size(0) <= self._policy1.size(
        #    0)), "seq_len of target should be less than or equal to seq_len of policy"
        assert (target.size(0) <= self._policy2.size(
            0)), "seq_len of target should be less than or equal to seq_len of policy"
        assert (target.size(1) == 1), "batch of target should be 1"

        if self._reset_optimizer:
            self._optimizer1.load_state_dict(self._optimizer1_orig_state)
            self._optimizer2.load_state_dict(self._optimizer2_orig_state)

        print("action inference")

        # Perform action inference cycles
        for i in range(self._inference_cycles):
            self._optimizer1.zero_grad()
            self._optimizer2.zero_grad()

            # Forward pass
            output = self.predict(x, block, actionnum, orientationnum, first_block, current_block=current_block,
                                  testing=testing)

            # Compute loss
            if first_block:
                loss = self._criterion(output.view(1, self.block_sz), target.view(1, self.block_sz),
                                       blocks=current_blocks_in_game, test=False, first=True)
            else:
                loss = self._criterion(output.view(1, self.block_sz), target.view(1, self.block_sz),
                                       blocks=current_blocks_in_game, test=True)
            loss = loss.mean()

            # Backward pass
            self._model.zero_grad()
            loss.backward()
            if not first_block:
                self._optimizer1.step()
            self._optimizer2.step()
            self._policy1.data, self._policy2.data = self.clamp(self._policy1.data, self._policy2.data)

        # Policy have been optimized; this optimized policy is now propagated
        # once more in forward direction in order to generate the final output
        # to be returned
        with torch.no_grad():
            output = self.predict(x, block, actionnum, orientationnum, first_block, current_block=current_block,
                                  testing=testing)

        # Save optimized policy to return
        policy1 = self._policy1.clone()
        policy2 = self._policy2.clone()

        action = torch.cat((actionnum, policy1), dim=-1)
        action = torch.cat((action, orientationnum), dim=-1)
        action = torch.cat((action, policy2), dim=-1)

        return action, output
