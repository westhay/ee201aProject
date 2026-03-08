def simulator_simulate(thermal_resistances, temperatures):
    """
    Simulates the temperature distribution in a system based on given thermal resistances and temperature inputs.

    Parameters:
    thermal_resistances (list): A list of thermal resistance values (in degrees Celsius per watt).
    temperatures (list): A list of initial temperatures (in degrees Celsius).

    Returns:
    list: A list of computed temperatures after heat transfer.
    """  

    if len(thermal_resistances) != len(temperatures) - 1:
        raise ValueError("The number of thermal resistances must be one less than the number of temperature nodes.")

    # Simple heat transfer calculation
    computed_temperatures = [temperatures[0]]

    for i in range(len(thermal_resistances)):
        # Calculate temperature at the next node using basic thermal balance
        temp_change = (temperatures[i] - computed_temperatures[-1]) / thermal_resistances[i]
        new_temp = computed_temperatures[-1] + temp_change
        computed_temperatures.append(new_temp)

    return computed_temperatures
