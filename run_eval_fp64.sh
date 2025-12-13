#!/bin/bash

# Script to run eval_fp64.py for each matrix name in valid_matrix_set.csv starting from the 1st matrix
# Record all completed matrices (converged or reached max iterations) with progress updates every 10 matrices

# Check if the CSV file exists
if [ ! -f "valid_matrix_set.csv" ]; then
    echo "Error: valid_matrix_set.csv not found!"
    exit 1
fi

# File to store all completed matrices (converged or reached max iterations)
COMPLETED_FILE="completed_matrices.txt"
> "$COMPLETED_FILE"  # Clear the file

# Counter for batch updates
MATRIX_COUNTER=0

# Function to run evaluation
run_evaluation() {
    local matrix_name="$1"
    local temp_file=$(mktemp)
    local converged=false
    local completed=false

    echo "Running evaluation for matrix: $matrix_name"

    # Run python script and capture output
    python eval/eval_fp64.py --matrix_name "$matrix_name" > "$temp_file" 2>&1
    local exit_code=$?

    # Check if converged by examining the output
    if grep -q "当前迭代是否收敛: True" "$temp_file"; then
        converged=true
        completed=true

        # Check if converged before reaching max iterations
        if ! grep -q "Reached maximum iterations" "$temp_file"; then
            # Add the complete matrix information to valid_matrix_set.csv
            # Ensure we start on a new line
            echo >> valid_matrix_set.csv
            awk -F',' -v name="$matrix_name" '$3 == name {print $0}' matrix_set.csv >> valid_matrix_set.csv
            echo "Matrix $matrix_name added to valid_matrix_set.csv (converged before max iterations)"
        fi
    elif grep -q "All evaluations completed!" "$temp_file" || [ "$exit_code" -eq 0 ]; then
        # Matrix completed (either converged or reached max iterations)
        completed=true
    fi

    # Display the output
    cat "$temp_file"

    # Record completion status
    if [ "$completed" = true ]; then
        if [ "$converged" = true ]; then
            echo "$matrix_name,converged" >> "$COMPLETED_FILE"
            echo "Matrix $matrix_name converged successfully"
        else
            echo "$matrix_name,max_iterations" >> "$COMPLETED_FILE"
            echo "Matrix $matrix_name reached max iterations without convergence"
        fi
    else
        echo "Matrix $matrix_name evaluation failed or was terminated"
    fi

    # Clean up
    rm -f "$temp_file"

    echo "Completed evaluation for matrix: $matrix_name"
    echo "----------------------------------------"

    return $exit_code
}

# Skip the header line, then extract the Name column (3rd column)
# Start from the 1st matrix
tail -n +2 valid_matrix_set.csv | awk -F',' '{print $3}' | while read -r matrix_name; do
    run_evaluation "$matrix_name"

    # Increment counter and update progress every 10 matrices
    MATRIX_COUNTER=$((MATRIX_COUNTER + 1))
    if [ $((MATRIX_COUNTER % 10)) -eq 0 ]; then
        echo "Progress update: $MATRIX_COUNTER matrices processed so far"
        echo "Current completed matrices count: $(wc -l < "$COMPLETED_FILE")"
        echo "----------------------------------------"
    fi
done

echo "All evaluations completed!"
echo "Total matrices processed: $MATRIX_COUNTER"
echo "Completed matrices saved to: $COMPLETED_FILE"
echo "Summary:"
echo "- Converged matrices: $(grep ",converged$" "$COMPLETED_FILE" | wc -l)"
echo "- Max iterations reached: $(grep ",max_iterations$" "$COMPLETED_FILE" | wc -l)"
