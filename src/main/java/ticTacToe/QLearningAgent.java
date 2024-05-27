package ticTacToe;

import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

/**
 * A Q-Learning agent with a Q-Table, i.e. a table of Q-Values. This table is
 * implemented in the {@link QTable} class.
 * 
 * The methods to implement are: (1) {@link QLearningAgent#train} (2)
 * {@link QLearningAgent#extractPolicy}
 * 
 * Your agent acts in a {@link TTTEnvironment} which provides the method
 * {@link TTTEnvironment#executeMove} which returns an {@link Outcome} object,
 * in other words an [s,a,r,s']: source state, action taken, reward received,
 * and the target state after the opponent has played their move. You may
 * want/need to edit {@link TTTEnvironment} - but you probably won't need to.
 * 
 * @author ae187
 */

public class QLearningAgent extends Agent {

	/**
	 * The learning rate, between 0 and 1.
	 */
	double alpha = 0.1;

	/**
	 * The number of episodes to train for
	 */
	int numEpisodes = 10000;

	/**
	 * The discount factor (gamma)
	 */
	double discount = 0.9;

	/**
	 * The epsilon in the epsilon greedy policy used during training.
	 */
	double epsilon = 0.1;

	/**
	 * This is the Q-Table. To get an value for an (s,a) pair, i.e. a (game, move)
	 * pair.
	 * 
	 */

	QTable qTable = new QTable();

	/**
	 * This is the Reinforcement Learning environment that this agent will interact
	 * with when it is training. By default, the opponent is the random agent which
	 * should make your q learning agent learn the same policy as your value
	 * iteration and policy iteration agents.
	 */
	TTTEnvironment env = new TTTEnvironment();

	/**
	 * Construct a Q-Learning agent that learns from interactions with
	 * {@code opponent}.
	 * 
	 * @param opponent     the opponent agent that this Q-Learning agent will
	 *                     interact with to learn.
	 * @param learningRate This is the rate at which the agent learns. Alpha from
	 *                     your lectures.
	 * @param numEpisodes  The number of episodes (games) to train for
	 */
	public QLearningAgent(Agent opponent, double learningRate, int numEpisodes, double discount) {
		env = new TTTEnvironment(opponent);
		this.alpha = learningRate;
		this.numEpisodes = numEpisodes;
		this.discount = discount;
		initQTable();
		train();
	}

	/**
	 * Initialises all valid q-values -- Q(g,m) -- to 0.
	 * 
	 */

	protected void initQTable() {
		List<Game> allGames = Game.generateAllValidGames('X');// all valid games where it is X's turn, or it's terminal.
		for (Game g : allGames) {
			List<Move> moves = g.getPossibleMoves();
			for (Move m : moves) {
				this.qTable.addQValue(g, m, 0.0);
				// System.out.println("initing q value. Game:"+g);
				// System.out.println("Move:"+m);
			}

		}

	}

	/**
	 * Uses default parameters for the opponent (a RandomAgent) and the learning
	 * rate (0.2). Use other constructor to set these manually.
	 */
	public QLearningAgent() {
		this(new RandomAgent(), 0.1, 50000, 0.9);

	}

	/**
	 * Implement this method. It should play {@code this.numEpisodes} episodes of
	 * Tic-Tac-Toe with the TTTEnvironment, updating q-values according to the
	 * Q-Learning algorithm as required. The agent should play according to an
	 * epsilon-greedy policy where with the probability {@code epsilon} the agent
	 * explores, and with probability {@code 1-epsilon}, it exploits.
	 * 
	 * At the end of this method you should always call the {@code extractPolicy()}
	 * method to extract the policy from the learned q-values. This is currently
	 * done for you on the last line of the method.
	 */

	public void train() {
		// Initializing the training process
		for (int episode = 0; episode < numEpisodes; episode++) {

			// Resetting the environment for the new episode
			env.reset();

			// Getting the current game state
			Game currentGame = env.getCurrentGameState();

			// Checking if the game has reached a terminal state
			while (!env.isTerminal()) {	
				Move selectedMove = null;

				// Deciding whether to explore or exploit
				if (new Random().nextDouble() < epsilon) { // exploration

					// Fetching the list of possible moves
					List<Move> possibleMoves = env.getPossibleMoves();

					// Selecting a random move for exploration
					selectedMove = possibleMoves.get(new Random().nextInt(possibleMoves.size()));

				} else { // exploitation
					double highestQValue = Double.NEGATIVE_INFINITY;

					// Iterating over possible moves to find the one with the highest Q-value
					for (Move candidateMove : env.getPossibleMoves()) {

						// Getting the Q-value for the candidate move
						double candidateQValue = qTable.getQValue(currentGame, candidateMove);

						// Updating the highest Q-value and selected move if the candidate's Q-value is
						// higher
						if (candidateQValue > highestQValue) {
							highestQValue = candidateQValue;
							selectedMove = candidateMove;
						}
					}
				}

				try {
					// Executing the selected move and getting the outcome
					Outcome outcome = env.executeMove(selectedMove);

					// Creating a final copy of the current game state 
					final Game currentGameFinal = currentGame;

					// Updating the maxQValue
					double maxQValue = Double.NEGATIVE_INFINITY;
					for (Move move : currentGameFinal.getPossibleMoves()) {
						double qValue = qTable.getQValue(currentGameFinal, move);
						maxQValue = Math.max(qValue, maxQValue);
					}

					// Calculating the sample for the Q-value update
					double sample;
					if (currentGameFinal.isTerminal()) {
						sample = outcome.localReward;
					} else {
						for (Move move : currentGameFinal.getPossibleMoves()) {
							double qValue = qTable.getQValue(currentGameFinal, move);
							if (qValue > maxQValue) {
								maxQValue = qValue;
							}
						}
						sample = outcome.localReward + discount * maxQValue;
					}

					// Calculating the running average
					double RunningAvg = (1 - alpha) * qTable.getQValue(outcome.s, outcome.move) + alpha * sample;

					// Updating the Q-value in the Q-table
					qTable.addQValue(outcome.s, outcome.move, RunningAvg);

					// Updating the current game state
					currentGame = outcome.sPrime;
				} catch (IllegalMoveException e) {

					// Printing the Exception message
					System.out.print("Illegal Move!!!!!!!!");
				}
			}
		}
		// --------------------------------------------------------
		// you shouldn't need to delete the following lines of code.
		this.policy = extractPolicy();
		if (this.policy == null) {
			System.out.println("Unimplemented methods! First implement the train() & extractPolicy methods");
			// System.exit(1);
		}

	}

	/**
	 * Implement this method. It should use the q-values in the {@code qTable} to
	 * extract a policy and return it.
	 *
	 * @return the policy currently inherent in the QTable
	 */
	public Policy extractPolicy() {
	    // Creating a new Optimal Policy object
	    Policy p = new Policy();

	    // For each 'state' in the key set of 'valueFunction'
	    for (Game state : qTable.keySet()) {

	        // Checking if the current state is not terminal
	        if (!state.isTerminal()) {

	            // Initializing 'maxQvalue' to negative infinity
	            double maxQvalue = Double.NEGATIVE_INFINITY;

	            // Initializing 'bestMove' to null
	            Move bestMove = null;

	            // For each possible 'move' in the current 'state'
	            for (Move move : state.getPossibleMoves()) {

	                // Initializing 'qValue' to 0
	                double qValue = qTable.getQValue(state, move);

	                // If 'qValue' is greater than 'maxQ', updating 'maxQ' to be 'qValue' and updating
	                // 'bestMove' to be the current 'move'
	                if (qValue > maxQvalue) {
	                    maxQvalue = qValue;
	                    bestMove = move;
	                }
	            }

	            // Updating the policy of the current 'state' to be 'bestMove'
	            p.policy.put(state, bestMove);
	        }
	    }

	    // Returning the Optimal Policy
	    return p;
	}


	public static void main(String a[]) throws IllegalMoveException {
		// Test method to play your agent against a human agent (yourself).
		QLearningAgent agent = new QLearningAgent();
		HumanAgent d = new HumanAgent();
		Game g = new Game(agent, d, d);
		g.playOut();

	}

}
