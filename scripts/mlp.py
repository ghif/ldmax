import jax
import jax.numpy as jnp
from flax import nnx
import optax


# 1. Define the Neural Network (Looks just like a PyTorch nn.Module!)
class SimpleMLP(nnx.Module):
    def __init__(
        self, in_features: int, hidden_features: int, out_features: int, *, rngs: nnx.Rngs
    ):
        self.layer1 = nnx.Linear(in_features, hidden_features, rngs=rngs)
        self.layer2 = nnx.Linear(hidden_features, hidden_features, rngs=rngs)
        self.out = nnx.Linear(hidden_features, out_features, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = nnx.relu(self.layer1(x))
        x = nnx.relu(self.layer2(x))
        return self.out(x)


# 2. Instantiate Model and Optax Optimizer
# We provide an explicit RNG stream for deterministic weight initialization
rngs = nnx.Rngs(params=42)
model = SimpleMLP(in_features=1, hidden_features=32, out_features=1, rngs=rngs)
optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=0.01), wrt=nnx.Param)


# 3. Define the JIT-Compiled Training Step
# @nnx.jit compiles the forward pass, gradient calculation, and weight update into a single XLA kernel
@nnx.jit
def train_step(
    model: SimpleMLP, optimizer: nnx.Optimizer, x_batch: jax.Array, y_batch: jax.Array
) -> jax.Array:
    def loss_fn(model: SimpleMLP):
        predictions = model(x_batch)
        return jnp.mean((predictions - y_batch) ** 2)

    # Compute loss and gradients simultaneously with autodiff
    loss, grads = nnx.value_and_grad(loss_fn)(model)

    # Update model parameters in-place (just like optimizer.step() in PyTorch)
    optimizer.update(model, grads)
    return loss


# 4. Generate Synthetic Data and Run the Training Loop
key = jax.random.key(0)
x_train = jax.random.uniform(key, shape=(256, 1), minval=-2.0, maxval=2.0)
y_train = jnp.sin(x_train) + 0.1 * jax.random.normal(key, shape=(256, 1))

for epoch in range(1, 2000):
    loss = train_step(model, optimizer, x_train, y_train)
    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss:.6f}")
