# MCTS implementation notes

`chess/scr/mcts.py` exposes `mcts_search(position, model=None, ...)`. It returns legal `Move` objects mapped to normalized visit probabilities; `return_root=True` also returns the reusable tree. `best_move(probabilities, root)` breaks equal visits by value, then prior. A terminal root returns an empty distribution; a zero-simulation budget returns legal priors.

## Search loop

1. **Select:** try each legal move once in prior order, then choose the largest PUCT score, `Q + c_puct * P * sqrt(parent_visits + 1) / (1 + edge_visits)`.
2. **Expand:** generate legal moves and attach policy priors. The default gives modest extra weight to captures, promotions, and castling. A neural model supplies logits masked to legal actions.
3. **Evaluate:** checkmate is -1 for the player to move, draws are 0, and nonterminal positions use either bounded material/placement evaluation or the network value head. This implementation uses leaf evaluation rather than random rollouts to the end of every game.
4. **Back up:** negate the value at each ply and update visits and accumulated value. Each edge stores value from its **parent player's** perspective, so both players maximize their own result.

Neural leaves are evaluated in batches. Virtual visits/losses discourage duplicate selection while a batch is pending; reservations are replaced by actual values after evaluation. A reused root must match the board, player, castling/en-passant rights, halfmove counter, repetition history, and evaluator identity. Search never changes the caller's position. Updating weights on the same model object requires discarding its old search tree.

Self-play adds fresh Dirichlet noise at each root and samples the improved policy using `p ** (1 / temperature)`. Temperature zero chooses the most visited move. Training retains original normalized visits, independent of move-sampling temperature.

## Neural representation

The 20 input planes encode 12 piece planes, side to move, four castling rights, en-passant target, normalized halfmove clock, and current-position repetition count. This includes more rule state than the original encoder, but not the entire history of positions; the rules engine retains that history separately for exact draw detection.

The 4,480-action space preserves the original 4,096 source/destination indices for ordinary moves and queen promotions. Another 384 slots distinguish knight/bishop/rook promotions by color and source/destination file. Illegal actions are masked during search. Distinct promotion choices no longer overwrite one another in self-play targets.

Value targets are +1/-1 for a win/loss by the recorded player, and 0 for draws. Training also labels move-limit truncations as draws; this starting policy can bias training when games are cut short. Small smoke runs prove generation, backpropagation, and checkpoint IO work; they do not establish playing strength.

## Next experiments

Measure the built-in evaluator at fixed simulation and time budgets before replacing it with a learned evaluator. Add a held-out tactical set and paired-color matches against the previous version. Track win/draw/loss rates, nodes per second, and time usage. Only promote a checkpoint after those comparisons; a randomly initialized network can play much worse than the handcrafted evaluator.

Later work includes better positional evaluation, tactical leaf handling, transposition-aware search with correct draw histories, richer historical inputs, smaller policy heads, checkpoint gating, and resumable replay/optimizer state.
