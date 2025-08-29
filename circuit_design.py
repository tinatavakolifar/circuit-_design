import schemdraw
import schemdraw.logic as logic
import schemdraw.elements as elm
import csv, os
import math


def intro():
    """
    Print an introduction and prompt the user for the CSV file path.

    Returns:
        str: The path to the user-provided CSV file.

    Significance:
        This function introduces the circuit design project to the user and collects
        the file path for the required truth or state table, which is essential as
        the first step in the design process.
    """
    print(
        '''CIRCUIT DESIGN.

        This project can visualize circuits by analyzing CSV data.

        Please provide the table carefully:
        - If it is a sequential circuit, enter its State Table.
        - If it is a combinational circuit, enter its Truth Table.
        
        Required format: (copy-paste for fewer errors).
        - State Table: first row must be "Qt, input, Qt+1, output"
                       (Qt and Qt+1 can have more than 1 items)

                       second row must be each item's name (i.e. "A,B,...,X,A,B,...,Y"
                       in which "A,B,..." are bit states, "X" is the input and "Y" the output)

        - Truth Table: first row must be "input, output"
                       second row must be each item's name (i.e. "X,Y,...,F" in which
                       "X,Y,..." are inputs and F the output or "Cin,X,Y,...,Cout,F" in which
                       "Cin" is the first item of the inputs and "Cout, F" are the outputs.)
        '''
        )
    file_path = input('Please enter CSV file path: ').strip()
    return file_path

def get_data(file_path):
    """
    Parse the provided CSV file and extract circuit data.

    Args:
        file_path (str): Path to the CSV file containing the truth or state table.

    Returns:
        tuple: (
            second_row (list of str): The subheader row (signal names),
            minterm_list (list): List of minterms (for combinational circuits),
            is_combinational (bool): True if the circuit is combinational,
            has_carry (bool): True if the circuit uses carry bits,
            subheader (list of str): List of all variable names,
            entry_data (int): Number of input variables,
            state_trans_list (list): List of state transitions (for sequential circuits),
            output_list (list): Indices of minterms where output is 1,
            Cout_list (list): Indices of minterms where carry out is 1,
            qt (list): State variable names,
            qt1 (list): Next state variable names,
            input_sqn (list): Input variable names (sequential),
            output_sqn (list): Output variable names (sequential)
        )

    Significance:
        This function is responsible for reading and validating the user's CSV file,
        determining whether the circuit is combinational or sequential, and extracting
        all relevant data for further processing, such as minimization and visualization.
    """
    minterm_list_temp = []
    has_carry = False
    entry_data = half = 0
    state_trans_list = []
    is_combinational = None
    output_list = []
    Cout_list = []
    qt = []
    qt1 = []
    input_sqn = []
    output_sqn = []

    while True:
        try:
            with open(file_path, newline='') as csvfile:
                reader = csv.reader(csvfile)
                first_row = next(reader)
                header = [h.strip().lower() for h in first_row]
                second_row = next(reader)
                subheader = [s.strip().lower() for s in second_row]
                data_rows = list(reader)

                # Flexible detection
                header_keywords = set(header)
                # Detect type based on keywords in header
                if any("qt" in h for h in header) and any("qt+1" in h for h in header):
                    is_combinational = False
                    break
                elif "input" in header and "output" in header:
                    is_combinational = True
                    break
                else:
                    # fallback: guess based on structure
                    if len(header) > 2 and "output" in header[-1]:
                        is_combinational = True
                        break
                    raise ValueError("Could not determine table type (truth/state).")
        except (FileNotFoundError, StopIteration, ValueError) as error:
            print(f'Error: {error}')
            file_path = input('Please enter CSV file path again: ').strip()

    if is_combinational:
        if "cin" in subheader and "cout" in subheader:
            has_carry = True
        entry_data = len(subheader) - (2 if has_carry else 1)
        minterm = 0
        for row in data_rows:
            if len(row) != len(subheader):
                raise ValueError(f"CSV row length {len(row)} does not match subheader length {len(subheader)}: {row}")
            if has_carry:
                cin_idx = subheader.index("cin") if "cin" in subheader else None
                cout_idx = subheader.index("cout") if "cout" in subheader else None
                row_inputs = row[:cout_idx] if cout_idx is not None else row[:-2]
                if int(row[-2]) == 1:
                    Cout_list.append(minterm)
                if int(row[-1]) == 1:
                    output_list.append(minterm)
                    minterm_list_temp.append((minterm, row_inputs))
            else:
                if int(row[-1]) == 1:
                    output_list.append(minterm)
                    minterm_list_temp.append((minterm, row[:-1]))
            minterm += 1
        minterm_list = sorted(minterm_list_temp, key=lambda x: x[0])
    else:
        minterm_list = []
        num_columns = len(subheader)
        half = num_columns // 2
        qt = subheader[:half]
        qt1 = subheader[half:half+len(qt)]
        remaining = subheader[half+len(qt):]
        input_sqn = [c for c in remaining if c not in qt1]
        output_sqn = [c for c in remaining if c not in input_sqn]
        # Parse sequential data into state_trans_list
        state_trans_list.clear()
        for row in data_rows:
            if len(row) != len(subheader):
                raise ValueError(f"CSV row length {len(row)} does not match subheader length {len(subheader)}: {row}")
            qt_vals = row[:half]
            qt1_vals = row[half:half+len(qt)]
            remaining_vals = row[half+len(qt):]
            inp_vals = []
            out_vals = []
            for idx, c in enumerate(remaining):
                if c in input_sqn:
                    inp_vals.append(remaining_vals[idx])
                elif c in output_sqn:
                    out_vals.append(remaining_vals[idx])
            state_trans_list.append((qt_vals, inp_vals, qt1_vals, out_vals))

    return (
        second_row,
        minterm_list,
        is_combinational,
        has_carry,
        subheader,
        entry_data,
        state_trans_list,
        output_list,
        Cout_list,
        qt, qt1,
        input_sqn,
        output_sqn
    )

def QM_grouping(entry_data, minterm_list):
    """
    Group minterms by the number of ones in their binary representation (Quine-McCluskey grouping).

    Args:
        entry_data (int): Number of input variables.
        minterm_list (list): List of minterms (index, input vector).

    Returns:
        tuple: (
            minterms_binary (list): Groups of minterms by ones-count,
            minterm_list (list): The original minterm list
        )

    Significance:
        This is the first step in the Quine-McCluskey algorithm, which is a systematic
        way of minimizing Boolean functions by grouping minterms according to the number
        of ones in their binary representation.
    """
    minterms_binary = []
    same_ones_list = []

    # Sort minterm_list by number of ones in binary representation
    minterms_sorted = sorted(minterm_list, key=lambda x: format(x[0], f'0{entry_data}b').count('1'))
    previous = None
    for i in range(len(minterms_sorted)):
        binary = format(minterms_sorted[i][0], f'0{entry_data}b')
        count_ones = binary.count('1')
        if i == 0:
            same_ones_list.append(([minterms_sorted[i][0]], binary))
            previous = count_ones
        elif count_ones == previous:
            same_ones_list.append(([minterms_sorted[i][0]], binary))
        else:
            minterms_binary.append(same_ones_list)
            same_ones_list = []
            previous = count_ones
            same_ones_list.append(([minterms_sorted[i][0]], binary))
    minterms_binary.append(same_ones_list)  # add the last group
    return minterms_binary, minterms_sorted

def QM_PI(minterms_binary):
    """
    Find Prime Implicants (PIs) using the Quine-McCluskey algorithm.

    Args:
        minterms_binary (list): Groups of minterms by ones-count.

    Returns:
        list: List of prime implicants as tuples (minterm indices, binary pattern).

    Significance:
        This function performs the iterative combination of minterms to extract all
        prime implicants, which are essential for optimal Boolean minimization.
    """
    current_groups = minterms_binary
    PIs = []

    while current_groups:
        next_groups = []
        combined_this_round = []
        combined_terms = set()

        for i in range(len(current_groups) - 1):
            this_list = current_groups[i]
            next_list = current_groups[i + 1]
            same_ones_list = []

            for j in this_list:
                combined = False
                for k in next_list:
                    j_indices = j[0] if isinstance(j[0], list) else [j[0]]
                    k_indices = k[0] if isinstance(k[0], list) else [k[0]]
                    # Compare all pairs of indices between j and k
                    found = False
                    for ji in j_indices:
                        for ki in k_indices:
                            if bin(ji ^ ki).count('1') == 1:
                                diff = [bit for bit in range(len(j[1])) if j[1][bit] != k[1][bit]][0]
                                new_binary = j[1][:diff] + '_' + j[1][diff+1:]
                                new_indices = sorted(set(j_indices + k_indices))
                                new = (new_indices, new_binary)
                                if new not in same_ones_list:
                                    same_ones_list.append(new)
                                combined = True
                                found = True
                                break
                        if found:
                            break
                if not combined and j not in PIs:
                    PIs.append(j)
            if same_ones_list:
                next_groups.append(same_ones_list)
                combined_this_round.append(same_ones_list)

        # Add all terms from current_groups not already in PIs
        for group in current_groups:
            for term in group:
                if term not in PIs:
                    PIs.append(term)

        if not combined_this_round:
            break
        current_groups = next_groups

    # Remove duplicates
    pi = []
    for item in PIs:
        if item not in pi:
            pi.append(item)
    return pi

def QM_EPI(pi, minterm_list):
    """
    Find Essential Prime Implicants (EPIs) from the list of Prime Implicants.

    Args:
        pi (list): List of prime implicants (from QM_PI).
        minterm_list (list): List of minterms (index, input vector).

    Returns:
        list: List of essential prime implicant binary patterns.

    Significance:
        Identifying essential prime implicants is crucial for further simplifying
        the Boolean function and ensuring minimal logic implementation.
    """
    epi = []
    
    # coverage dict: key = minterm index, value = list of PIs that cover it
    coverage = {m_index: [] for m_index, _ in minterm_list}
    for m_index, *_ in minterm_list:  # loop through decimal minterm
        for pi_index, binary in pi:
            # Ensure pi_index is iterable before using 'in'
            if isinstance(pi_index, (tuple, list)) and m_index in pi_index:
                coverage[m_index].append(binary)
    
    for m_index, p in coverage.items():  # minterms covered by exactly one PI
        if len(p) == 1 and p[0] not in epi:
            epi.append(p[0])

    return epi

def state_diagram(state_trans_list, PDF_FOLDER, filename='state_diagram'):
    """
    Visualize sequential circuit state diagram using schemdraw.

    Args:
        state_trans_list (list): List of state transitions as tuples (Qt, input, Qt+1, output).
        PDF_FOLDER (str): Folder to save the output PNG.
        filename (str): Filename prefix for the PNG.

    Returns:
        None
    """
    # Fill state_trans_list if not already filled (for robustness)
    if not state_trans_list or not isinstance(state_trans_list[0], tuple):
        raise ValueError("state_trans_list is empty or not in the correct format.")

    # Compute positions for states (simple horizontal layout)
    states = list({''.join(qt) for qt, _, _, _ in state_trans_list}.union(
                  {''.join(qt1) for _, _, qt1, _ in state_trans_list}))
    n = len(states)
    spacing = 4
    positions = {state: (i * spacing, 0) for i, state in enumerate(states)}

    with schemdraw.Drawing() as d:
        d.config(fontsize=14)
        state_nodes = {}
        # Draw state nodes
        for state, (x, y) in positions.items():
            state_nodes[state] = d.add(elm.Circle().at((x, y)).label(state, 'center'))

        # Draw transitions (including self-loops)
        for qt, inp, qt1, outp in state_trans_list:
            cur = ''.join(qt)
            nxt = ''.join(qt1)
            label = f"{''.join(inp)}/{''.join(outp)}"
            start = positions[cur]
            end = positions[nxt]
            import math
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            dist = math.hypot(dx, dy)
            r = 1.0  # approximate circle radius in schemdraw units
            if dist == 0:
                # Self-loop: draw an arc above the state
                arc_center = (start[0], start[1]+r)
                arc = d.add(elm.Arc(radius=1.5, theta1=60, theta2=320).at(arc_center).lw(1.2).color('black'))
                # Arrow head for self-loop
                arrow_x = start[0] + 1.5*math.cos(math.radians(60))
                arrow_y = start[1] + 1.5*math.sin(math.radians(60))
                d.add(elm.Line().at((arrow_x, arrow_y)).to((arrow_x+0.3, arrow_y)).arrow().label(label, loc='top'))
            else:
                # Draw a curved or straight arrow between different states
                ux = dx / dist
                uy = dy / dist
                start_edge = (start[0] + ux * r, start[1] + uy * r)
                end_edge = (end[0] - ux * r, end[1] - uy * r)
                # If there are multiple transitions between the same states, curve the line
                # (Optional: for now, just draw straight)
                d.add(elm.Line().at(start_edge).to(end_edge).arrow().label(label, loc='top'))

        # Save drawing
        outpath = os.path.join(PDF_FOLDER, f"{filename}.png")
        d.save(outpath)
        print(f"State diagram saved as '{outpath}'")


# Helper function to remove constant columns
def remove_constant_columns(data_rows, subheader):
    """
    Remove columns that are constant (all 0s or all 1s) from consideration.

    Args:
        data_rows (list of list): Rows of CSV data (excluding headers).
        subheader (list of str): Column names.

    Returns:
        list: Filtered subheader with constant columns removed.
    """
    active_cols = []
    for i, name in enumerate(subheader):
        values = [row[i] for row in data_rows if i < len(row)]
        if not values:
            continue
        if len(set(values)) == 1:  # all 0s or all 1s
            continue
        active_cols.append(name)
    return active_cols

def special(
        is_combinational,
        has_carry,
        output_list,
        Cout_list,
        subheader,
        minterm_list,
        state_table_shape=None
        ):
    """
    Improved special circuit type detection based on header structure.
    - Correctly detects Half-Adder even if Cin exists but is always zero.
    - Removes Full-Adder misdetection for Half-Adder CSVs.
    - Detection based on number of input/output columns and names.
    """
    special_type = None

    if is_combinational:
        # Filter subheader for meaningful inputs and outputs
        outputs_keywords = ['s', 'sum', 'f', 'cout', 'y']
        inputs_filtered = [col for col in subheader if col.lower() not in outputs_keywords]
        outputs_filtered = [col for col in subheader if col.lower() in outputs_keywords]

        num_inputs = len(inputs_filtered)
        num_outputs = len(outputs_filtered)

        # Half-Adder detection: 2 inputs, 2 outputs
        if num_inputs == 2 and num_outputs == 2:
            special_type = 'Half-Adder'
        # Full-Adder detection: 3 inputs, 2 outputs
        elif num_inputs == 3 and num_outputs == 2:
            special_type = 'Full-Adder'
        # Other special circuits as before, using num_inputs and num_outputs
        # 4-bit Adder: 5 inputs, 4 outputs
        elif num_inputs == 5 and num_outputs == 4:
            special_type = '4-bit Adder'
        elif num_inputs == 4 and num_outputs == 5:
            special_type = 'BCD Adder'
        # Subtractor, Multiplier, Comparator, Decoder, Encoder, MUX, etc. as before
        elif subheader:
            if len(subheader) == 4 and set(subheader[-2:]) & {'d','bout'}:
                special_type = 'Half Subtractor'
            elif len(subheader) == 5 and set(subheader[-2:]) & {'d','bout'}:
                special_type = 'Full Subtractor'
            elif len(subheader) == 7 and any('sub' in s for s in subheader):
                special_type = '4-bit Subtractor'
            # multiplier
            if len(subheader) == 6:
                outs = subheader[-4:]
                if all(any(p in o for p in ('p0','p1','p2','p3')) for o in outs):
                    special_type = '2-bit Multiplier'
            # comparator
            if ('a=b' in subheader) or ('a>b' in subheader) or ('a<b' in subheader):
                special_type = 'Comparator'
            # decoder
            n_inputs = len(subheader) - (2 if has_carry else 1)
            n_outputs = num_outputs
            if minterm_list and n_outputs == 2 ** n_inputs:
                special_type = f'{n_inputs}-to-{2**n_inputs} Decoder'
            # encoder
            if minterm_list is not None and n_inputs > n_outputs and n_outputs > 0:
                special_type = 'Encoder'
            # multiplexer
            if (
                'sel' in ''.join(subheader).lower() or
                's' in ''.join(subheader).lower()
                ):
                special_type = 'Multiplexer (MUX)'
        # fallback
        if special_type is None:
            if subheader and len(subheader) == 4 and not has_carry:
                special_type = 'Simple 2-input Combinational (unknown)'
    else:
        # Sequential detection remains unchanged
        if state_table_shape:
            num_states, num_inputs, num_next, num_outputs = state_table_shape
            if num_states == 1 and num_next == 1 and num_outputs == 1:
                if num_inputs == 1: special_type = 'D-Latch'
                if num_inputs == 2: special_type = 'D Flip-Flop (D-FF)'
                if num_inputs == 2: special_type = 'JK Flip-Flop (JK-FF)'
                if num_inputs == 1: special_type = 'T Flip-Flop (T-FF)'
            elif num_states == num_next:
                if num_inputs == num_outputs and num_states > 1:
                    special_type = f'{num_states}-bit Register'
                if num_inputs == 1 and num_outputs == num_states:
                    special_type = f'{num_states}-bit Counter Register'
            if num_states > 1 and num_inputs == 1:
                if num_outputs == 1:
                    special_type = f'{num_states}-bit SISO Shift Register'
                if num_outputs == num_states:
                    special_type = f'{num_states}-bit SIPO Shift Register'
            if num_states > 1:
                if num_inputs == num_states:
                    if num_outputs == 1:
                        special_type = f'{num_states}-bit PISO Shift Register'
                    if num_outputs == num_states:
                        special_type = f'{num_states}-bit PIPO Shift Register'
        else:
            if subheader:
                sh = [s.lower() for s in subheader]
                if any('d' in s for s in sh) and any('q' in s for s in sh):
                    special_type = 'D Flip-Flop (D-FF)'
                elif any('jk' in s for s in sh):
                    special_type = 'JK Flip-Flop (JK-FF)'
                elif any('t' in s for s in sh):
                    special_type = 'T Flip-Flop (T-FF)'
                elif any('reg' in s for s in sh):
                    special_type = 'Register'
                elif any('count' in s for s in sh):
                    special_type = 'Counter Register'
                elif any('shift' in s for s in sh):
                    special_type = 'Shift Register'
    return special_type

def visualize(
        PDF_FOLDER,
        subheader,
        has_carry,
        epi, pi,
        output_list,
        Cout_list,
        minterm_list,
        is_combinational,
        special_type,
        qt, qt1,
        input_sqn,
        output_sqn,
        state_trans_list,
        filename_prefix='circuit_visual'
        ):
    """
    Visualize combinational circuit using standard logic gates via schemdraw.

    Only combinational circuits are supported in this version.
    """
    if not is_combinational:
        print("Sequential circuit visualization not implemented in schemdraw version.")
        return

    inputs = subheader[:-2] if has_carry else subheader[:-1]
    outputs = subheader[-2:] if has_carry else subheader[-1:]

    for kind, terms, color, suffix in [
        ('hazard_free', pi, 'lightblue', '_hazard_free'),
        ('simplified', epi, 'lightgreen', '_simplified')
    ]:
        with schemdraw.Drawing() as d:
            d.config(fontsize=14)
            input_nodes = {}
            y_spacing = 2.0
            # Place input dots vertically
            for i, inp in enumerate(inputs):
                input_nodes[inp] = d.add(logic.Dot().at((0, -i * y_spacing))).label(inp, 'left')

            and_gates = []
            and_gate_inputs = []
            # Place AND gates horizontally, one per term
            for idx, term_tuple in enumerate(terms):
                if isinstance(term_tuple, tuple):
                    term = term_tuple[1]  # binary pattern
                else:
                    term = term_tuple  # already a string
                # Each AND gate is placed at increasing x
                x_and = 4 + idx * 3
                y_and = -len(inputs) * y_spacing / 2
                and_gate = d.add(logic.And().at((x_and, y_and)).right())
                and_gates.append(and_gate)
                and_inputs = []
                # Connect each input to the AND gate according to the term
                for i, bit in enumerate(term):
                    inp_name = inputs[i]
                    if bit == '1':
                        # Direct line from input to AND input
                        d.add(logic.Line().at(input_nodes[inp_name].end).to(and_gate.in1 if i == 0 else and_gate.in2))
                        and_inputs.append((inp_name, False))
                    elif bit == '0':
                        # NOT gate between input and AND input
                        not_x = (input_nodes[inp_name].end[0] + x_and) / 2
                        not_y = input_nodes[inp_name].end[1]
                        not_gate = d.add(logic.Not().at((not_x, not_y)).right())
                        d.add(logic.Line().at(input_nodes[inp_name].end).to(not_gate.in1))
                        d.add(logic.Line().at(not_gate.out).to(and_gate.in1 if i == 0 else and_gate.in2))
                        and_inputs.append((inp_name, True))
                    # '_' = don’t care, skip
                and_gate_inputs.append(and_inputs)

            # Place OR gates for each output
            for out_idx, out in enumerate(outputs):
                # Place OR gate to the right of all AND gates
                x_or = 4 + len(terms) * 3 + 3
                y_or = -out_idx * 3.0
                # Determine number of inputs for the OR gate
                num_or_inputs = len(and_gates)
                # Use the correct OR gate for number of inputs
                if num_or_inputs <= 2:
                    or_gate = d.add(logic.Or().at((x_or, y_or)).right())
                    or_inputs = [or_gate.in1, or_gate.in2]
                else:
                    or_gate = d.add(logic.Or(n=num_or_inputs).at((x_or, y_or)).right())
                    # schemdraw.logic.Or(n=...) provides .inputs property
                    or_inputs = getattr(or_gate, "inputs", [])
                    # Fallback: generate connection points if not present
                    if not or_inputs or len(or_inputs) < num_or_inputs:
                        # Place input points vertically spaced
                        or_inputs = []
                        for i in range(num_or_inputs):
                            or_inputs.append((x_or, y_or - 0.7*(i - (num_or_inputs-1)/2)))
                # Output dot
                out_dot = d.add(logic.Dot().at(or_gate.out).right()).label(out, 'right')
                # Connect AND gates to OR gate
                or_input_idx = 0
                for idx, and_gate in enumerate(and_gates):
                    # For circuits with carry, only connect relevant ANDs to Cout or F
                    if has_carry:
                        minterm_idx = minterm_list[idx][0] if (idx < len(minterm_list) and isinstance(minterm_list[idx], tuple)) else None
                        if out.lower() == 'cout':
                            if minterm_idx in Cout_list:
                                target = or_inputs[or_input_idx] if or_input_idx < len(or_inputs) else or_gate.in1
                                d.add(logic.Line().at(and_gate.out).to(target))
                                or_input_idx += 1
                        else:
                            if minterm_idx in output_list:
                                target = or_inputs[or_input_idx] if or_input_idx < len(or_inputs) else or_gate.in1
                                d.add(logic.Line().at(and_gate.out).to(target))
                                or_input_idx += 1
                    else:
                        target = or_inputs[or_input_idx] if or_input_idx < len(or_inputs) else or_gate.in1
                        d.add(logic.Line().at(and_gate.out).to(target))
                        or_input_idx += 1

            filename = os.path.join(PDF_FOLDER, f'{filename_prefix}{suffix}')
            d.save(f'{filename}.png')
            print(f"Saved {kind} circuit as '{filename}.png'")

    return inputs, outputs, Cout_list, output_list

def main():
    """
    Main entry point for the circuit design tool.

    Prompts the user for a CSV file, processes the data, determines the circuit type,
    minimizes logic if possible, and generates visualizations.

    Returns:
        None

    Significance:
        Orchestrates all steps in the circuit design workflow: input, analysis,
        minimization, recognition of standard types, and visualization.
    """
    PDF_FOLDER = "output_pdfs"
    os.makedirs(PDF_FOLDER, exist_ok=True)

    file_path = intro()
    (
        second_row,
        minterm_list,
        is_combinational,
        has_carry,
        subheader,
        entry_data,
        state_trans_list,
        output_list,
        Cout_list,
        qt, qt1,
        input_sqn,
        output_sqn
    ) = get_data(file_path)

    # Filter constant columns from subheader for better special-type detection
    if is_combinational:
        data_rows_filtered = [inputs for _, inputs in minterm_list]
        subheader = remove_constant_columns(data_rows_filtered, subheader)
    else:
        data_rows_filtered = [list(qt) + list(inp) + list(qt1) + list(outp) for qt, inp, qt1, outp in state_trans_list]
        subheader = remove_constant_columns(data_rows_filtered, subheader)

    # Decide algorithm based on type and size
    if is_combinational:
        if entry_data <= 8:
            minterms_binary, minterm_list_sorted = QM_grouping(entry_data, minterm_list)
            pi = QM_PI(minterms_binary)
            epi = QM_EPI(pi, minterm_list_sorted)
            special_type = special(
                is_combinational,
                has_carry,
                output_list,
                Cout_list,
                subheader,
                minterm_list
                )

            inputs, outputs, Cout_used, out_used = visualize(
                PDF_FOLDER,
                subheader, has_carry, epi, pi, output_list, Cout_list,
                minterm_list, is_combinational, special_type,
                qt, qt1, input_sqn, output_sqn, state_trans_list
            )
        else:
            print("Espresso algorithm placeholder (not implemented).")
            special_type = special(
                is_combinational,
                has_carry,
                output_list,
                Cout_list,
                subheader,
                minterm_list
                )
    else: # Sequential:
        # Add state diagram visualization
        if state_trans_list is not None and len(state_trans_list) > 0:
            state_diagram(state_trans_list, PDF_FOLDER)
            if entry_data <= 5:
                # Table/Partition Refinement logic (call special and visualize)
                special_type = special(
                    is_combinational,
                    has_carry,
                    output_list,
                    Cout_list,
                    subheader,
                    minterm_list
                    )
                visualize(
                    PDF_FOLDER,
                    subheader, has_carry, [], [], output_list, Cout_list,
                    minterm_list, is_combinational, special_type,
                    qt, qt1, input_sqn, output_sqn, state_trans_list
                )
            elif 5 < entry_data <= 20:
                print("Sequential: BDD/State Encoding placeholder.")
                special_type = special(
                    is_combinational,
                    has_carry,
                    output_list,
                    Cout_list,
                    subheader,
                    minterm_list
                    )
                visualize(
                    PDF_FOLDER,
                    subheader, has_carry, [], [], output_list, Cout_list,
                    minterm_list, is_combinational, special_type,
                    qt, qt1, input_sqn, output_sqn, state_trans_list
                )
            else:
                print("Sequential: Heuristic/Approximate method placeholder.")
                special_type = special(
                    is_combinational,
                    has_carry,
                    output_list,
                    Cout_list,
                    subheader,
                    minterm_list
                    )
                visualize(
                    PDF_FOLDER,
                    subheader, has_carry, [], [], output_list, Cout_list,
                    minterm_list, is_combinational, special_type,
                    qt, qt1, input_sqn, output_sqn, state_trans_list
                )

    print(f"Detected circuit type: {special_type}")

if __name__ == "__main__":
    main()

# Add gui later

# Example CSVs for Testing


# Minimal truth table for a Half Adder (combinational)
# Inputs: Cin, A, B; Outputs: Cout, S
# Save as a .csv file and use for testing.
EXAMPLE_HALF_ADDER = """
input,output
Cin,A,B,Cout,S
0,0,0,0,0
0,0,1,0,1
0,1,0,0,1
0,1,1,1,0
"""
# Minimal state table for JK Flip-Flop (sequential)
# Inputs: J, K; States: Q(t), Q(t+1); Output: Qnext
# Save as a .csv file and use for testing.
EXAMPLE_JK_FF = """
qt,input,qt+1,output
Q,J,K,Qnext
0,0,0,0,0
0,0,1,0,0
0,1,0,1,1
0,1,1,1,0
1,0,0,1,1
1,0,1,1,0
1,1,0,0,1
1,1,1,0,0
"""
# Minimal truth table for a 2-to-1 Multiplexer (combinational)
# Inputs: S, D0, D1; Output: Y
# Save as a .csv file and use for testing.
EXAMPLE_MUX = """
input,output
S,D0,D1,Y
0,0,0,0
0,0,1,0
0,1,0,1
0,1,1,1
1,0,0,0
1,0,1,1
1,1,0,0
1,1,1,1
"""

# Usage:
# Copy the content of any EXAMPLE_* variable into a .csv file
# and provide its path to the program as input.
