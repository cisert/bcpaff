"""
© 2023, ETH Zurich
"""


def find_cps_and_paths():
    return """
2 # Topology analysis
2 # Search CPs from nuclear positions
3 # Search CPs from midpoint of atom pairs
8 # Generating the paths connecting (3,-3) and (3,-1) CPs
-10 # Return to main menu
"""


def keep_only_intermolecular(atom_idxs1, atom_idxs2):
    return f"""
2 # Topology analysis
-5 # Modify or print detail or export paths, or plot property along a path
8 # Only retain bond paths (and corresponding CPs) connecting two specific molecular fragments while remove all other bond paths and BCPs
{atom_idxs1} # atoms belonging to first fragment
{atom_idxs2} # atoms belonging to second fragment
y # remove corresponding BCPs
0 # Return
-10 # Return to main menu
"""


def remove_all_but_bcps():
    return """
2 # Topology analysis
-4 # Modify or export CPs (critical points)
2 # Delete some CPs
3 # Delete all (3,-3) CPs
5 # Delete all (3,+1) CPs
6 # Delete all (3,+3) CPs
0 # Return
0 # Return
-10 # Return to main menu
"""


def save_paths():
    return """
2 # Topology analysis
-5 # Modify or print detail or export paths, or plot property along a path
4 # Save points of all paths to paths.txt in current folder
0 # Return
-10 # Return to main menu
"""


def save_cps(include_esp=True):
    props = 0 if include_esp else -1
    return f"""
2 # Topology analysis
-4 # Modify or export CPs (critical points)
4 # Save CPs to CPs.txt in current folder
0 # Return
7 # Show real space function values at specific CP or all CPs
{props} # 0 for all properties, -1 to skip ESP --> makes it faster
-10 # Return to main menu
"""


def save_cps_to_pdb():
    return """
2 # Topology analysis
-4 # Modify or export CPs (critical points)    
6 # Export CPs as CPs.pdb file in current folder
0 # Return
-10 # Return to main menu
"""


def save_paths_to_pdb():
    return """
2 # Topology analysis
-5 # Modify or print detail or export paths, or plot property along a path  
6 # Export paths as paths.pdb file in current folder
0 # Return
-10 # Return to main menu
"""


def exit_gracefully():
    return "q # Exit program gracefully"
