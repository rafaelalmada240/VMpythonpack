Code as implemented in the article *"Healing Regimes of Microscopic Wounds in the Vertex Model of Cell Tissues"* by Almada RF, et al, Phys Rev E (2025). This code was originally implemented in Linux, and for Windows, it can be run with WSL (although we recommend it to be run natively as it can be quite slow). Running the code in Windows should be done cautiously as Pool works differently in that OS. Soon, we will provide a Windows-compatible of the code, with minimal changes to the current implementation.

The code available is to be used as follows:

- **makenetwork2** - generates a network with a wound at the center of user-defined size. The network revelant data is stored as txt files in a folder named 'tissues/tissue(user defined number)/size(wound size)/'
- **runwh** - runs the simulation on the user generated network according to parameters defined in the file *inputrwh.txt*.
- **makenetmovie** - after the simulation if the option to generate movie data was chosen, it is possible to generate a video with the output. 
