# AI-Agents

## Files you can edit

ValueIterationAgent.java	A Value Iteration agent for solving the Tic-Tac-Toe game with an assumed MDP model.<br/>
PolicyIterationAgent.java	A Policy Iteration agent for solving the Tic-Tac-Toe game with an assumed MDP model.<br/>
QLearningAgent.java	A q-learner, Reinforcement Learning agent for the Tic-Tac-Toe game.<br/>

## Files you should read & use but shouldn’t need to edit

Game.java	The 3x3 Tic-Tac-Toe game implementation.<br/>
TTTMDP.java	Defines the Tic-Tac-Toe MDP model<br/>
TTTEnvironment.java	Defines the Tic-Tac-Toe Reinforcement Learning environment.<br/>
Agent.java	Abstract class defining a general agent, which other agents subclass.<br/>
HumanAgent.java	Defines a human agent that uses the command line to ask the user for the next move.<br/>
RandomAgent.java	Tic-Tac-Toe agent that plays randomly according to a RandomPolicy.<br/>
Move.java	Defines a Tic-Tac-Toe game move.<br/>
Outcome.java	A transition outcome tuple (s,a,r,s’)<br/>
Policy.java	An abstract class defining a policy – you should subclass this to define your own policies.<br/>
TransitionProb.java	A tuple containing an Outcome object and a probability of the Outcome occurring.<br/>
RandomPolicy.java	A subclass of policy – it’s a random policy used by a RandomAgent instance.
