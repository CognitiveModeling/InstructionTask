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
            self, model, policy2, optimizer2, inference_cycles=30,
            criterion=torch.nn.MSELoss(), reset_optimizer=True,
            policy_handler=lambda x: x):

        #assert (len(policy1.shape) ==
        #        3), "policy should be of shape (seq_len, batch, input_size)"
        assert (len(policy2.shape) ==
                3), "policy should be of shape (seq_len, batch, input_size)"
        #assert (policy1.size(1) == 1), "batch of policy should be 1"
        assert (policy2.size(1) == 1), "batch of policy should be 1"

        self._model = model
        #self._policy1 = policy1
        self._policy2 = policy2
        #self._policy1.requires_grad = True
        self._policy2.requires_grad = True
        self._inference_cycles = inference_cycles
        self._criterion = criterion
        #self._optimizer1 = optimizer1
        self._optimizer2 = optimizer2
        self._reset_optimizer = reset_optimizer
        if self._reset_optimizer:
            #self._optimizer1_orig_state = optimizer1.state_dict()
            self._optimizer2_orig_state = optimizer2.state_dict()
        self._policy_handler = policy_handler

    def recreate_onehot(self, blocks):
        for i in range(self._policy2.size(0)):
            one = torch.argmax(blocks[i][0])
            u = blocks[i][0]

            for guess in range(0, len(blocks[i][0])):
                u[guess] = 0.0

            u[one] = 1.0

            blocks[i][0] = u

        return blocks

    def predict(self, x, blocks, state):
        """Predict into future.

        Predict observations given the current policy as well as an initial
        input and hidden state, and a context.

        Parameters
        ----------
        x : torch.Tensor
            Initial input.
        state : torch.Tensor or tuple
            Initial hidden (and cell) state of the network.
        context : torch.Tensor
            Context activations over the whole prediction.

        Returns
        -------
        outputs : list
            Result of the prediction.
        states : list of torch.Tensor or list of tuple
            Hidden (and cell) states of the model corresponding to the
            predicted outputs.

        """

        outputs = []
        states = []
        # Forward pass over policy
        for t in range(self._policy2.size(0)):
            #block = self._policy1[t:t+1]
            position = self._policy2[t:t+1]
            block = blocks[t:t+1]
            #inp = torch.cat((context[t:t+1], self._policy[t:t+1], x), dim=2)
            x, state = self._model(x, block, position, state)
            outputs.append(x)
            states.append(state)
        return outputs, states

    def action_inference(self, x, blocks, state, target):
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

        assert (len(x.shape) ==
                3), "x should be of shape (seq_len, batch, input_size)"
        assert (x.size(0) == 1), "seq_len of x should be 1"
        assert (x.size(1) == 1), "batch of x should be 1"

        assert (len(target.shape) ==
                3), "target should be of shape (seq_len, batch, output_size)"
        #assert (target.size(0) <= self._policy1.size(
        #    0)), "seq_len of target should be less than or equal to seq_len of policy"
        assert (target.size(0) <= self._policy2.size(
            0)), "seq_len of target should be less than or equal to seq_len of policy"
        assert (target.size(1) == 1), "batch of target should be 1"

        if self._reset_optimizer:
            #self._optimizer1.load_state_dict(self._optimizer1_orig_state)
            self._optimizer2.load_state_dict(self._optimizer2_orig_state)

        # Perform action inference cycles
        for _ in range(self._inference_cycles):
            #self._optimizer1.zero_grad()
            self._optimizer2.zero_grad()

            # Forward pass
            outputs, _ = self.predict(x, blocks, state)

            # Compute loss
            start = len(self._policy2) - len(target)
            loss = self._criterion(outputs[start], target)

            # Backward pass
            loss.backward()
            #self._optimizer1.step()
            self._optimizer2.step()

            # Operations on the data are not tracked
            #self._policy1.data = self._policy_handler(self._policy1.data)
            self._policy2.data = self._policy_handler(self._policy2.data)

            #with torch.no_grad():
            #    self._policy1 = self.recreate_onehot(self._policy1)

        # Policy have been optimized; this optimized policy is now propagated
        # once more in forward direction in order to generate the final output
        # to be returned
        with torch.no_grad():
            outputs, states = self.predict(x, blocks, state)

        # Save optimized policy to return
        #policy1 = self._policy1.clone()
        policy2 = self._policy2.clone()

        # Shift policy, last entry is duplicated
        with torch.no_grad():
            #self._policy1[:-1] = self._policy1[1:].clone()
            self._policy2[:-1] = self._policy2[1:].clone()

        return policy2, outputs, states
