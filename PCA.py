import numpy as np

class PCA:
    """Principal component analysis (PCA).
    Parameters
    ----------
    n_components : int
        Number of principal components to use.
    whiten : bool, default=False
        When true, the output of transformed features is divided by the
        square root of the explained variance.
    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> pca = PCA(n_components=2)
    >>> pca.fit(X)
    >>> pca.transform(X)
    >>> array([[ 1.38340578,  0.2935787 ],
               [ 2.22189802, -0.25133484],
               [ 3.6053038 ,  0.04224385],
               [-1.38340578, -0.2935787 ],
               [-2.22189802,  0.25133484],
               [-3.6053038 , -0.04224385]])
    """
    def __init__(self, n_components: int, whiten: bool = False) -> None:
        self.n_components = n_components
        self.whiten = whiten
        self.selected_components = None
        self.mean = None 
                   
    def fit(self, X: np.ndarray) -> None:
        """Fit the model with X.
        Parameters
        ----------
        X : a numpy array with dimensions (n_samples, n_features)
        """        
        #Step 1: Find the mean, and center the data
        self.mean = X.mean(axis=0)
        X = X - self.mean
        
        #Step2:  Find the Covariance
        cov = np.cov(X, rowvar=False)

        #Step 3: Apply SVD and choose the components, make the hermitian argument True.
        U, S, Vh = np.linalg.svd(cov, hermitian=True)
        self.selected_components = Vh
        # choose the singular values of diagnal matrix
        self.explained_variance = S
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X with the fitted model.
        Parameters
        ----------
        X : a numpy array with dimensions (n_samples, n_features)
        
        Returns
        -------
        X_transformed: a numpy array with dimensions (n_samples, n_components)
        """
        # Center the data
        X = X - self.mean
        
        # Step 4: Choose and transform the features
        X_transformed =np.dot(X, self.selected_components[:self.n_components].T)
        if self.whiten:
            # Normalize the transform features
            X_transformed /= np.sqrt(self.explained_variance[:self.n_components])
        return X_transformed
        