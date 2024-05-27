package ticTacToe;

import java.util.HashMap;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.Set;

/**
 * A policy iteration agent. You should implement the following methods: (1)
 * {@link PolicyIterationAgent#evaluatePolicy}: this is the policy evaluation
 * step from your lectures (2) {@link PolicyIterationAgent#improvePolicy}: this
 * is the policy improvement step from your lectures (3)
 * {@link PolicyIterationAgent#train}: this is a method that should
 * runs/alternate (1) and (2) until convergence.
 * 
 * NOTE: there are two types of convergence involved in Policy Iteration:
 * Convergence of the Values of the current policy, and Convergence of the
 * current policy to the optimal policy. The former happens when the values of
 * the current policy no longer improve by much (i.e. the maximum improvement is
 * less than some small delta). The latter happens when the policy improvement
 * step no longer updates the policy, i.e. the current policy is already
 * optimal. The algorithm should stop when this happens.
 * 
 * @author ae187
 *
 */
public class PolicyIterationAgent extends Agent {

	/**
	 * This map is used to store the values of states according to the current
	 * policy (policy evaluation).
	 */
	HashMap<Game, Double> policyValues = new HashMap<Game, Double>();

	/**
	 * This stores the current policy as a map from {@link Game}s to {@link Move}.
	 */
	HashMap<Game, Move> curPolicy = new HashMap<Game, Move>();

	double discount = 0.9;

	/**
	 * The mdp model used, see {@link TTTMDP}
	 */
	TTTMDP mdp;

	/**
	 * loads the policy from file if one exists. Policies should be stored in .pol
	 * files directly under the project folder.
	 */
	public PolicyIterationAgent() {
		super();
		this.mdp = new TTTMDP();
		initValues();
		initRandomPolicy();
		train();

	}

	/**
	 * Use this constructor to initialise your agent with an existing policy
	 * 
	 * @param p
	 */
	public PolicyIterationAgent(Policy p) {
		super(p);

	}

	/**
	 * Use this constructor to initialise a learning agent with default MDP
	 * paramters (rewards, transitions, etc) as specified in {@link TTTMDP}
	 * 
	 * @param discountFactor
	 */
	public PolicyIterationAgent(double discountFactor) {

		this.discount = discountFactor;
		this.mdp = new TTTMDP();
		initValues();
		initRandomPolicy();
		train();
	}

	/**
	 * Use this constructor to set the various parameters of the Tic-Tac-Toe MDP
	 * 
	 * @param discountFactor
	 * @param winningReward
	 * @param losingReward
	 * @param livingReward
	 * @param drawReward
	 */
	public PolicyIterationAgent(double discountFactor, double winningReward, double losingReward, double livingReward,
			double drawReward) {
		this.discount = discountFactor;
		this.mdp = new TTTMDP(winningReward, losingReward, livingReward, drawReward);
		initValues();
		initRandomPolicy();
		train();
	}

	/**
	 * Initialises the {@link #policyValues} map, and sets the initial value of all
	 * states to 0 (V0 under some policy pi ({@link #curPolicy} from the lectures).
	 * Uses {@link Game#inverseHash} and {@link Game#generateAllValidGames(char)} to
	 * do this.
	 * 
	 */
	public void initValues() {
		List<Game> allGames = Game.generateAllValidGames('X');// all valid games where it is X's turn, or it's terminal.
		for (Game g : allGames)
			this.policyValues.put(g, 0.0);

	}

	/**
	 * You should implement this method to initially generate a random policy, i.e.
	 * fill the {@link #curPolicy} for every state. Take care that the moves you
	 * choose for each state ARE VALID. You can use the
	 * {@link Game#getPossibleMoves()} method to get a list of valid moves and
	 * choose randomly between them.
	 */
	
	public void initRandomPolicy() {

		// Getting all states from policy values
		Set<Game> states = policyValues.keySet();
		// Creating a new random object
		Random rand = new Random();

		// Iterating through each state
		for (Game state : states) {

			// Checking if the state is not terminal
			if (!state.isTerminal()) {

				// Getting the list of possible moves
				List<Move> posMoves = state.getPossibleMoves();

				// Checking if the list of possible moves is not empty
				if (!posMoves.isEmpty()) {
					
					// Selecting a random move
					Move randMove = posMoves.get(rand.nextInt(posMoves.size()));
					
					// Updating the policy with the random move
					this.curPolicy.put(state, randMove);
				}
			}
		}
	}

	/**
	 * Performs policy evaluation steps until the maximum change in values is less
	 * than {@code delta}, in other words until the values under the currrent policy
	 * converge. After running this method, the
	 * {@link PolicyIterationAgent#policyValues} map should contain the values of
	 * each reachable state under the current policy. You should use the
	 * {@link TTTMDP} {@link PolicyIterationAgent#mdp} provided to do this.
	 *
	 * @param delta
	 */
	protected void evaluatePolicy(double delta) {
	    // Initializing maxQVal to a very low number
	    double maxQVal;

	    // Iterating until convergence
	    do {
	        // Resetting maxQVal to negative infinity
	        maxQVal = Double.NEGATIVE_INFINITY;

	        // Iterating through all games in the current policy
	        for (Game state : curPolicy.keySet()) {
	            // Initializing qVal to 0
	            double qVal = 0;

	            // Checking if the game is not in a terminal state
	            if (!state.isTerminal()) {
	            	
	                // Policy Iteration Bellman's Equation for non-terminal states
	                for (TransitionProb tr : mdp.generateTransitions(state, curPolicy.get(state))) {
	                    qVal += tr.prob * (tr.outcome.localReward + (discount * policyValues.get(tr.outcome.sPrime)));
	                }
	            }

	            // Getting the old Q value for the current game
	            double oldQVal = policyValues.get(state);

	            // Updating the Q value for the current game
	            policyValues.put(state, qVal);

	            // Calculating the maximum Q value
	            maxQVal = Math.max(maxQVal, Math.abs(oldQVal - qVal));
	        }
	        
	    // Continuing the loop as long as the maximum change in Q value is greater than delta
	    } while (maxQVal > delta);
	}

	/**
	 * This method should be run AFTER the
	 * {@link PolicyIterationAgent#evaluatePolicy} train method to improve the
	 * current policy according to {@link PolicyIterationAgent#policyValues}. You
	 * will need to do a single step of expectimax from each game (state) key in
	 * {@link PolicyIterationAgent#curPolicy} to look for a move/action that
	 * potentially improves the current policy.
	 * 
	 * @return true if the policy improved. Returns false if there was no
	 *         improvement, i.e. the policy already returned the optimal actions.
	 */

	// Initializing the flag to check if the policy has improved
	protected boolean improvePolicy() {
		// Checking if policy is improved
	    boolean hasImproved = false;

	    // Iterating over all games in the current policy
	    for (Game state : curPolicy.keySet()) {

	        // Checking if the current state is not terminal
	        if (!state.isTerminal()) {

	            // Initializing 'maxQ' to negative infinity
	            double maxQ = Double.NEGATIVE_INFINITY;

	            // Initializing 'bestMove' to null
	            Move bestMove = null; 

	            // Iterating over all possible moves for the current game
	            for (Move move : state.getPossibleMoves()) {

	                // Initializing 'sum' to 0
	                double sum = 0;

	                // Policy Iteration Bellman's Equation
	                for (TransitionProb t : mdp.generateTransitions(state, move))
	                    sum += t.prob * (t.outcome.localReward + (discount * policyValues.get(t.outcome.sPrime)));

	                // If 'sum' is greater than 'maxQ'
	                if (sum > maxQ) {

	                    // Updating 'maxQ' to be 'sum'
	                    maxQ = sum;

	                    // Updating 'bestMove' to be the current 'move'
	                    bestMove = move;
	                }
	            }

	            // Checking if the optimal move is different from the current policy for the
	            // game
	            if (!(Objects.equals(bestMove, curPolicy.get(state)))) {

	                // Updating the current policy with the optimal move
	                curPolicy.put(state, bestMove);

	                // Indicating that the policy has improved
	                hasImproved = true;
	            }
	        }
	    }

	    // Returning the improved policy value
	    return hasImproved;
	}


	/**
	 * The (convergence) delta
	 */
	double delta = 0.1;

	/**
	 * This method should perform policy evaluation and policy improvement steps
	 * until convergence (i.e. until the policy no longer changes), and so uses your
	 * {@link PolicyIterationAgent#evaluatePolicy} and
	 * {@link PolicyIterationAgent#improvePolicy} methods.
	 */
	public void train() {
		
	    // Initializing the values
	    initValues();

	    // Initializing random policies
	    initRandomPolicy();

	    // Initializing a hash-map to store all the old policies
	    HashMap<Game, Move> prePolicy;

	    do {
	        // Evaluating the policy
	        evaluatePolicy(delta);

	        // Storing the current policy before improvement
	        prePolicy = new HashMap<>(this.curPolicy);

	        // Improving the policy
	        if (!improvePolicy()) {
	            break;
	        }

	        // Checking if the policies are the same, if so, breaking the loop
	        if (curPolicy.equals(prePolicy)) {
	            break;
	        }
	    } while (true);

	    // Creating a new policy
	    super.policy = new Policy();

	    // Setting the new policy as the current policy
	    super.policy.policy = curPolicy;
	}


	public static void main(String[] args) throws IllegalMoveException {
		/**
		 * Test code to run the Policy Iteration Agent agains a Human Agent.
		 */
		PolicyIterationAgent pi = new PolicyIterationAgent();

		HumanAgent h = new HumanAgent();

		Game g = new Game(pi, h, h);

		g.playOut();

	}

}
