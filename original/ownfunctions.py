import sim

def pleaseWait(deltaTime, clientID):
    return_code, _, _, _ = sim.simxCallScriptFunction(clientID, "ForThreadedScript", sim.sim_scripttype_childscript,
                                                      "wait_function", [deltaTime], [], [], bytearray(),
                                                      sim.simx_opmode_blocking)
    if return_code != sim.simx_return_ok:
        return return_code
