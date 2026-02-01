import numpy as np
import matplotlib.pyplot as plt


'''
notes
groups[M] = count
M is how many features are in each group (M features over M-1 dimensions)
count is how many groups of M there are


'''

class ProbeMatrix:


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

            for group in range(count):
                curr_rows = list(range(row_idx, row_idx+M)) #rows used currently by this group

                if nicely_laid_out:
                    temp = np.zeros((rank, D), dtype=float)
                    for x in range(rank):
                        temp[x, dim_idx+x] = 1.0
                else:
                    #Skipping for now
                    pass

                #Random linear combination of the prevoius rows
                for y in range(M):
                    coeffs = self.rng.normal(size=rank)
                    v = coeffs @ temp 
                    W[row_idx] = v
                    row_idx += 1

                self.metadata.append({"type": "group", "rows": curr_rows, "size": M, "rank": rank})

                if nicely_laid_out:
                    dim_idx+= rank
        
        # 3. Adding noise to matrix (sigma)

        self.W = W
        return W
    

def visualize(
        matrix: np.ndarray,
        cmap: str = "viridis",
        title: str = "Density Heatmap",
        show_colorbar: bool = True
    ):
        masked_matrix = np.ma.masked_where(matrix == 0, matrix)
        plt.figure()
        im = plt.imshow(masked_matrix, cmap=cmap, interpolation="nearest")
        plt.title(title)
        plt.xlabel("Columns")
        plt.ylabel("Rows")

        if show_colorbar:
            plt.colorbar(im)

        plt.tight_layout()
        plt.show()



matrix = np.array([
    [0, 2, 0, 5],
    [1, 3, 0, 0],
    [0, 0, 4, 7]
])

visualize(matrix)