package ticTacToe;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A Value Iteration Agent, only very partially implemented. The methods to
 * implement are: (1) {@link ValueIterationAgent#iterate} (2)
 * {@link ValueIterationAgent#extractPolicy}
 * 
 * You may also want/need to edit {@link ValueIterationAgent#train} - feel free
 * to do this, but you probably won't need to.
 * 
 * @author ae187
 *
 */
public class ValueIterationAgent extends Agent {

	/**
	 * This map is used to store the values of states
	 */
	Map<Game, Double> valueFunction = new HashMap<Game, Double>();

	/**
	 * the discount factor
	 */
	double discount = 0.9;

	/**
	 * the MDP model
	 */
	TTTMDP mdp = new TTTMDP();

	/**
	 * the number of iterations to perform - feel free to change this/try out
	 * different numbers of iterations
	 */
	int k = 10;

	/**
	 * This constructor trains the agent offline first and sets its policy
	 */
	public ValueIterationAgent() {
		super();
		mdp = new TTTMDP();
		this.discount = 0.9;
		initValues();
		train();
	}

	/**
	 * Use this constructor to initialise your agent with an existing policy
	 * 
	 * @param p
	 */
	public ValueIterationAgent(Policy p) {
		super(p);

	}

	public ValueIterationAgent(double discountFactor) {

		this.discount = discountFactor;
		mdp = new TTTMDP();
		initValues();
		train();
	}

	/**
	 * Initialises the {@link ValueIterationAgent#valueFunction} map, and sets the
	 * initial value of all states to 0 (V0 from the lectures). Uses
	 * {@link Game#inverseHash} and {@link Game#generateAllValidGames(char)} to do
	 * this.
	 * 
	 */
	public void initValues() {

		List<Game> allGames = Game.generateAllValidGames('X');// all valid games where it is X's turn, or it's terminal.
		for (Game g : allGames)
			this.valueFunction.put(g, 0.0);

	}

	public ValueIterationAgent(double discountFactor, double winReward, double loseReward, double livingReward,
			double drawReward) {
		this.discount = discountFactor;
		mdp = new TTTMDP(winReward, loseReward, livingReward, drawReward);
	}

	/**
	 * 
	 * 
	 * /* Performs {@link #k} value iteration steps. After running this method, the
	 * {@link ValueIterationAgent#valueFunction} map should contain the (current)
	 * values of each reachable state. You should use the {@link TTTMDP} provided to
	 * do this.
	 * 
	 *
	 */

	public void iterate() {
	    // Iterating 'k' times
	    for (int i = 0; i < k; i++) {
	        
	        // For each 'state' in the key set of 'valueFunction'
	        for (Game state : valueFunction.keySet()) {

	            // Checking if the current state is not terminal
	            if (!state.isTerminal()) {

	                // Initializing 'vNext' to negative infinity
	                double vNext = Double.NEGATIVE_INFINITY;

	                // For each possible 'move' in the current 'state'
	                for (Move m : state.getPossibleMoves()) {

	                    // Initializing 'qValue' to 0
	                    double qValue = 0;

	                    // Value Iteration Bellman's Equation
	                    for (TransitionProb t : mdp.generateTransitions(state, m)) {
	                        qValue += t.prob * (t.outcome.localReward + discount * valueFunction.get(t.outcome.sPrime));

	                        // Updating 'vNext' to be the maximum of 'vNext' and 'qValue'
	                        vNext = Math.max(vNext, qValue);
	                    }

	                    // Updating the value function of the current 'state' to be 'vNext'
	                    valueFunction.put(state, vNext);
	                }
	            }
	        }
	    }
	}

	

	/**
	 * This method should be run AFTER the train method to extract a policy
	 * according to {@link ValueIterationAgent#valueFunction} You will need to do a
	 * single step of expectimax from each game (state) key in
	 * {@link ValueIterationAgent#valueFunction} to extract a policy.
	 * 
	 * @return the policy according to {@link ValueIterationAgent#valueFunction}
	 */
	public Policy extractPolicy() {
	    // Creating a new Optimal Policy object
	    Policy p = new Policy();

	    // For each 'state' in the key set of 'valueFunction'
	    for (Game state : valueFunction.keySet()) {

	        // Checking if the current state is not terminal
	        if (!state.isTerminal()) {

	            // Initializing 'maxQvalue' to negative infinity
	            double maxQvalue = Double.NEGATIVE_INFINITY;

	            // Initializing 'bestMove' to null
	            Move bestMove = null;

	            // For each possible 'move' in the current 'state'
	            for (Move move : state.getPossibleMoves()) {

	                // Initializing 'qValue' to 0
	                double qValue = 0;

	                // Value Iteration Bellman's Equation
	                for (TransitionProb t : mdp.generateTransitions(state, move)) {
	                    qValue += t.prob * (t.outcome.localReward + (discount * valueFunction.get(t.outcome.sPrime)));
	                }

	                // If 'qValue' is greater than 'maxQ', Updating 'maxQ' to be 'qValue' and Updating 'bestMove' to be the current 'move'
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


	/**
	 * This method solves the mdp using your implementation of
	 * {@link ValueIterationAgent#extractPolicy} and
	 * {@link ValueIterationAgent#iterate}.
	 */
	public void train() {
		/**
		 * First run value iteration
		 */
		this.iterate();
		/**
		 * now extract policy from the values in
		 * {@link ValueIterationAgent#valueFunction} and set the agent's policy
		 * 
		 */

		super.policy = extractPolicy();

		if (this.policy == null) {
			System.out.println("Unimplemented methods! First implement the iterate() & extractPolicy() methods");
			// System.exit(1);
		}

	}

	public static void main(String a[]) throws IllegalMoveException {
		// Test method to play the agent against a human agent.
		ValueIterationAgent agent = new ValueIterationAgent();
		HumanAgent d = new HumanAgent();

		Game g = new Game(agent, d, d);
		g.playOut();

	}
}
