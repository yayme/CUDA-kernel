__global__ void naive_attention(
    float* Q,
    float* K,
    float* V,
    float* O,
    int seq_len,
    int head_dim
)
{
    // each thread should handle one output element O[i, j]
    int i = blockIdx.x*blockDim.x + threadIdx.x; // each row handles the full sequence for a single head. so we have to move along the row
    int j = blockIdx.y*blockDim.y + threadIdx.y; // each columns handles the full head dimension for a single token. so we have to move along the column
    if( i>= seq_len || j>=head_dim) return;

    // S[i,:]= Q[i, :] @ K.T;
    // S= S/sqrt(head_dim);
    // A= softmax(S, dim=-1);
    // O = A @ V;
    float  S[1024]; // assuming seq_len <= 1024
    // we need S[i,t]= dot (Q[i,:], K[t,:])
    for (int t=0 ; t<=seq_len; t++)
    {
        float dot = 0.0f;
        for(int s=0 ;s <= head_dim; s++)
        {
            
            dot+= Q[i*head_dim+s]*K[t*head_dim+s];
        }
        S[t]= dot/sqrt((float)head_dim);
    }
}