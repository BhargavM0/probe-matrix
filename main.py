import numpy as np
import matplotlib.pyplot as plt


'''
notes
groups[M] = count
M is how many features are in each group (M features over M-1 dimensions)
count is how many groups of M there are
'''

class ProbeMatrix:

    def __init__(self, seed= None):
        self.rng = np.random.default_rng(seed)

    def synthesize(
            self, 
            N: int, 
            D: int, 
            groups: dict[int, int],
            sigma: float = 0.0,
            nicely_laid_out: bool = True,
            rank_mode: str = "M-1"
        ) -> np.ndarray:

        #validating parameters
        if N <= 0 or D <= 0:
            raise ValueError("N and D must be positive and non-zero")
        if sigma < 0:
            raise ValueError("Sigma cannot be negative")
        
        #validating group characteristics
        for M, count in groups.items():
            if M < 2:
                raise ValueError("Dependent vectors must span at least 2 dimensions")
            if count < 0:
                raise ValueError("There cannot be negative counts of groups of dimensions M")
            
        
        #Finding independent and dependent rows
        dependent_rows = 0
        for M in groups:
            dependent_rows+= groups[M] * M

        if dependent_rows > N:
            raise ValueError(f"Not enough rows allocated. {dependent_rows} are needed but only {N} are allocated.")
        independent_rows = N - dependent_rows

        #Determining and validating the ranks per each group size
        ranks: dict[int, int] = {}
        for M in groups:
            if rank_mode == "M-1":
                rank = M-1
            else:
                #skipping custom ranks for now !!!
                pass

            if not (1 <= rank < M):
                raise ValueError(f"For group M = {M}, rank is either < 1 or is >= M. Rank is currently set to {rank}")
            ranks[M] = rank
        

        #Checking the feasibility of nicely_laid_out parameter
        if nicely_laid_out:
            dims_needed = 0
            for M in groups:
                dims_needed+= ranks[M] * groups[M] 
            dims_needed+=independent_rows
            if dims_needed > D:
                raise ValueError(f"Not enough dimensions to fit nicely_laid_out format. Need {dims_needed-D} more dimension(s) to be feasible.")

        #Creating empty weight matrix and metadata (keeping track of groups)
        W = np.zeros((N,D), dtype = float)
        self.metadata = []

        row_idx = 0
        dim_idx = 0

        # 1. Adding the independent rows
        for _ in range(independent_rows):
            W[row_idx, dim_idx] = 1.0
            self.metadata.append({"type": "independent", "rows": [row_idx], "size": 1, "rank": 1})
            row_idx+=1
            dim_idx+=1

        # 2. Adding the groups of rows
        for M in sorted(groups.keys()):
            count = int(groups[M])
            rank = int(ranks[M])

            for _ in range(count):
                curr_rows = list(range(row_idx, row_idx+M)) #rows used currently by this group
                basis_rows = []
                
                #The basis for the linear combination of the N vector
                if nicely_laid_out:
                    for i in range(rank):
                        v = np.zeros(D, dtype=float)
                        v[dim_idx + i] = 1.0
                        W[row_idx] = v
                        basis_rows.append(v.copy())
                        row_idx += 1
                else:
                    #Skipping for now
                    pass

                max_coeff = D + 1

                #Random linear combination of the prevoius rows
                for _ in range(M - rank):

                    coeffs = self.rng.integers(
                        low=-max_coeff,
                        high=max_coeff + 1,
                        size=rank
                    )

                    # prevent zero vector
                    while np.all(coeffs == 0):
                        coeffs = self.rng.integers(
                            low=-max_coeff,
                            high=max_coeff + 1,
                            size=rank
                        )

                    v = np.zeros(D, dtype=float)
                    for i in range(rank):
                        v += coeffs[i] * basis_rows[i]

                    W[row_idx] = v
                    row_idx += 1

                self.metadata.append({"type": "group", "rows": curr_rows, "size": M, "rank": rank})

                if nicely_laid_out:
                    dim_idx+= rank
        
        # 3. Adding noise to matrix (sigma)
        if sigma > 0:
            noise = self.rng.normal(loc=0.0, scale=sigma, size=W.shape)
            W = W + noise

        self.W = W
        return W



probe = ProbeMatrix()

#Example
'''
W = probe.synthesize(
    N=40,
    D=34,
    groups={3: 4, 5: 2},   
    sigma=0.0,
    nicely_laid_out=True
)
'''
probe.synthesize(
    N=5,
    D=4,
    groups={3: 1},   
    sigma=0.0,
    nicely_laid_out=True
)

print(probe.W)