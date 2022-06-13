# InstructionTask

Network that learns spatial relationships between building blocks in a CoppeliaSim simulation. The network predicts a block's resulting position, given a position where it is being released on the field. It takes into account positions and rotations of all other blocks in the game, recognizing when an intended position is impossible (i.e. colliding with the ground or another block).

Preparation:
The simulation is run using CoppeliaSim's legacy remote Python API (https://www.coppeliarobotics.com/helpFiles/en/legacyRemoteApiOverview.htm). 
When the remote API is set up, start up CoppeliaSim, load the scene "scenebase_various.ttt", then run 
simRemoteApi.start(19999)
in the CoppeliaSim console.

code file legend:

actioninference.py: holds the action inference class, which allows us to infer actions given a certain goal/target future state

attention_net.py: holds the network class

attentionnet_test.py: tests a trained network

create_arrangements.py: runs simulations on randomized block stacking tasks, saving action and resulted state for future training

create_samples.py: creates training samples out of selected saved training arrangements

create_test_arrangements.py: creates arrangements for testing

create_test_samples.py: creates samples for testing out of testing arrangements

shapes.py: holds class Shape, representing block objects in the simulation, as well as all necessary functions to interact with them
