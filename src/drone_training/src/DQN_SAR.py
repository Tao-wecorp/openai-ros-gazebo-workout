from DQN import DQN
import numpy as np

class DQN_SAR(DQN):
    def choose_action(self,state,epsilon):
        if np.random.random() <= epsilon:
            to_be_rescued = np.where(self.env.rescued==False)[0]
            rtb_action = len(self.env.survivors)
            to_be_rescued = np.append(to_be_rescued,rtb_action)
            return np.random.choice(to_be_rescued)
        else:
            if self.double_q:
                action_values = self.agent_network.predict(state) + self.target_network.predict(state)
                effective_action_values = np.full(action_values.shape,float("-inf"))
                effective_action_values = np.reshape(effective_action_values,(action_values.shape[1]))
                action_values = np.reshape(action_values,(action_values.shape[1]))
                tmp = np.copy(self.env.rescued)
                tmp = np.append(tmp,False) # Treat rtb as always not-rescued.
                effective_action_values[np.where(tmp==False)] = action_values[tmp==False]
                return np.argmax(effective_action_values)
            else:
                action_values = self.agent_network.predict(state)
                effective_action_values = np.full(action_values.shape,float("-inf"))
                effective_action_values = np.reshape(effective_action_values,(action_values.shape[1]))
                action_values = np.reshape(action_values,(action_values.shape[1]))
                tmp = np.copy(self.env.rescued)
                tmp = np.append(tmp,False) # Treat rtb as always not-rescued.
                effective_action_values[np.where(tmp==False)] = action_values[tmp==False]
                return np.argmax(effective_action_values)