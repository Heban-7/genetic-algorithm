; -----------------------------------------------------------------------------
; A Genetic Algorithm to Evolve a Binary String in MeTTa
;
; This script implements the core components of a genetic algorithm to find a
; target binary string. It demonstrates concepts like fitness calculation,
; selection, crossover, and mutation using a functional, term-rewriting style.
; -----------------------------------------------------------------------------


; =============================================================================
; 1. CONFIGURATION PARAMETERS
; =============================================================================
; We define the core parameters of our GA as global symbols.
; A symbol is defined to be equal to its value directly.

!(println! "Hello, world!")


(: TARGET_STRING (List Number))
(= TARGET_STRING (1 1 0 1 0 1 0 0 1 1))

(: CHROMOSOME_LENGTH Number)
(= CHROMOSOME_LENGTH (len TARGET_STRING))

(: POPULATION_SIZE Number)
(= POPULATION_SIZE 100)

(: MUTATION_RATE Number)
(= MUTATION_RATE 0.01)

(: CROSSOVER_RATE Number)
(= CROSSOVER_RATE 0.95)


; =============================================================================
; 2. UTILITY FUNCTIONS
; =============================================================================
; MeTTa is minimal, so we first define common list-processing utilities.

; (sum-list <list>) - Sums a list of numbers.
(: sum-list (-> (List Number) Number))
(= (sum-list $l)
   (if (empty? $l)
       0
       (+ (car $l) (sum-list (cdr $l)))))

; (map <function> <list>) - Applies a function to each element of a list.
(: map (-> Atom (List Atom) (List Atom)))
(= (map $f $l)
   (if (empty? $l)
       ()
       (cons ($f (car $l)) (map $f (cdr $l)))))

; (list-to-string <list>) - Converts a list of numbers to a string for printing.
(: list-to-string (-> (List Number) String))
(= (list-to-string $l)
   (if (empty? $l)
       ""
       (string-concat (number-to-string (car $l)) (list-to-string (cdr $l)))))

; =============================================================================
; 3. CORE GENETIC ALGORITHM COMPONENTS
; =============================================================================

; -----------------------------------------------------------------------------
; A. Population and Individual Creation
; -----------------------------------------------------------------------------

; (create-individual <length>) - Creates a random binary chromosome.
(: create-individual (-> Number (List Number)))
(= (create-individual $n)
   (if (== $n 0)
       ()
       (cons (rand-int 0 2) (create-individual (- $n 1)))))

; (create-population <size> <len>) - Creates an initial population of individuals.
(: create-population (-> Number Number (List (List Number))))
(= (create-population $size $len)
   (if (== $size 0)
       ()
       (cons (create-individual $len) (create-population (- $size 1) $len))))

; -----------------------------------------------------------------------------
; B. Fitness Evaluation
; -----------------------------------------------------------------------------

; (calculate-fitness <individual> <target>) - Counts matching bits.
(: calculate-fitness (-> (List Number) (List Number) Number))
(= (calculate-fitness $ind $target)
   (if (empty? $ind)
       0
       (+ (if (== (car $ind) (car $target)) 1 0)
          (calculate-fitness (cdr $ind) (cdr $target)))))

; -----------------------------------------------------------------------------
; C. Selection
; -----------------------------------------------------------------------------

; (roulette-select-recursive <pop> <fitness_scores> <pick>) - Helper for selection.
(: roulette-select-recursive (-> (List (List Number)) (List Number) Number (List Number)))
(= (roulette-select-recursive $pop $scores $pick)
   (let* ((current-fitness (car $scores))
           (current-individual (car $pop)))
         (if (<= $pick current-fitness)
             current-individual
             (roulette-select-recursive (cdr $pop) (cdr $scores) (- $pick current-fitness)))))

; (roulette-wheel-selection <pop> <fitness_scores>) - Selects one parent.
(: roulette-wheel-selection (-> (List (List Number)) (List Number) (List Number)))
(= (roulette-wheel-selection $pop $scores)
   (let* ((total-fitness (sum-list $scores))
           (pick (if (== total-fitness 0)
                     0 ; Avoid division by zero if all fitness is 0
                     (rand-float 0 total-fitness))))
         (roulette-select-recursive $pop $scores pick)))

; -----------------------------------------------------------------------------
; D. Crossover
; -----------------------------------------------------------------------------

; (single-point-crossover <p1> <p2>) - Performs single-point crossover.
(: single-point-crossover (-> (List Number) (List Number) (List (List Number))))
(= (single-point-crossover $p1 $p2)
   (if (> (rand-float 0 1) CROSSOVER_RATE)
       ($p1 $p2) ; Return parents unchanged if no crossover
       (let* ((point (rand-int 1 CHROMOSOME_LENGTH)))
             (let* ((child1 (concat (take $p1 point) (drop $p2 point)))
                    (child2 (concat (take $p2 point) (drop $p1 point))))
                   (child1 child2)))))

; -----------------------------------------------------------------------------
; E. Mutation
; -----------------------------------------------------------------------------

; (mutate <individual>) - Flips bits based on mutation rate.
(: mutate (-> (List Number) (List Number)))
(= (mutate $ind)
   (if (empty? $ind)
       ()
       (let* ((gene (car $ind))
              (mutated-gene (if (< (rand-float 0 1) MUTATION_RATE)
                                (- 1 gene) ; Flip the bit
                                gene)))
             (cons mutated-gene (mutate (cdr $ind))))))


; =============================================================================
; 4. MAIN EVOLUTIONARY LOOP
; =============================================================================

; (find-best-individual <pop>) - Finds the best individual and its fitness.
(: find-best-individual (-> (List (List Number)) (List Atom)))
(= (find-best-individual $pop)
   (let* ((fitness-fn (lambda ($i) (calculate-fitness $i TARGET_STRING)))
          (fitness-scores (map fitness-fn $pop)))
         (letrec* ((loop (lambda ($p $s $best_i $best_f)
                         (if (empty? $p)
                             ($best_i $best_f)
                             (let* ((current_i (car $p))
                                    (current_f (car $s)))
                                   (if (> current_f $best_f)
                                       (loop (cdr $p) (cdr $s) current_i current_f)
                                       (loop (cdr $p) (cdr $s) $best_i $best_f)))))))
                  (loop $pop $fitness-scores () -1))))


; (run-one-generation <pop>) - Creates the next generation of the population.
(: run-one-generation (-> (List (List Number)) (List (List Number))))
(= (run-one-generation $pop)
   (let* ((fitness-fn (lambda ($i) (calculate-fitness $i TARGET_STRING)))
          (fitness-scores (map fitness-fn $pop)))
         (letrec* ((build-next-gen (lambda ($n)
                                   (if (<= $n 0)
                                       ()
                                       (let* ((p1 (roulette-wheel-selection $pop $fitness-scores))
                                              (p2 (roulette-wheel-selection $pop $fitness-scores))
                                              (children (single-point-crossover p1 p2))
                                              (child1 (car children))
                                              (child2 (cadr children)))
                                             (cons (mutate child1)
                                                   (cons (mutate child2)
                                                         (build-next-gen (- $n 2)))))))))
                  (take (build-next-gen POPULATION_SIZE) POPULATION_SIZE))))


; (run-ga <pop> <gen> <max_gen>) - The main recursive loop for the GA.
(: run-ga (-> (List (List Number)) Number Number Nil))
(= (run-ga $pop $gen $max_gen)
   (if (>= $gen $max_gen)
       ; THEN branch: End of the run
       (let* ((_ (println! (string-concat "\nAlgorithm finished after " (number-to-string $max_gen) " generations."))))
             (let* ((result (find-best-individual $pop))
                    (best-ind (car result))
                    (best-fit (cadr result)))
                   (let* ((_ (println! (string-concat "Final Best Fitness: " (number-to-string best-fit)))))
                         (println! (string-concat "Final Best Individual: " (list-to-string best-ind))))))
       ; ELSE branch: Continue to next generation
       (let* ((result (find-best-individual $pop))
              (best-ind (car result))
              (best-fit (cadr result)))
             (let* ((_ (println! (string-concat "Generation " (number-to-string $gen)
                                                " | Best Fitness: " (number-to-string best-fit)
                                                " | Best Individual: " (list-to-string best-ind)))))
                   (if (== best-fit CHROMOSOME_LENGTH)
                       (println! "\nSolution found!")
                       (run-ga (run-one-generation $pop) (+ $gen 1) $max_gen))))))

; =============================================================================
; 5. EXECUTION
; =============================================================================

; Kick off the Genetic Algorithm
!(let* ((initial-population (create-population POPULATION_SIZE CHROMOSOME_LENGTH)))
      (run-ga initial-population 0 500))
