
(+);;
(*- : int -> int -> int = <fun> *)
(<=);;
(*- : 'a -> 'a -> bool = <fun>*)

let max (a,b) =
  if a < b then b else a;;

let max a b = 
    if a >  b then a else b;; (*curry*)

let max = function a ->
          function b -> if a > b then a else b (*version curry larga*)

let x, y = 2+1, 3*2;;  
       
let f n = n + 1, n - 1
let g m n = m + n , m * n
let g (m, n) = m + n, m * n

let quo x y = (*x >= 0, y > 0*)(*Calcular el cosiente*)
    if x < y then 0
    else 1 + quo (x-y) y
 
let rec rem x y = (*x >= 0, y > 0*) (*Calcular el resto*)
    if x < y then x
    else rem (x-y) y;     

let rec div x y = (*esto es un horror uwu*)
    if x < y then 0, x
    else 1 + fst (div (x-y) y), snd (div (x-y) y)
    
let rec div x y = (*esto es como tiene que ser uwu*)
    if x < y then 0, x
    else let q,r = div(x-y) y in 
    1 + q, r;;

let rec fib n = if <= 1 then n
    else fib(n-1) + fib(n-2);;
    
fib2 = int -> int*int;; 

let rec fib n = 
  if n <= 1 then n
  else fib(n-1) + fib(n-2);;
  
 let crono f x = 
  let t = Sys.time() in 
  f x;
  Sys.time() -. t;; 
  
  let crono f x = 
    let t = Sys.time() in 
    let _ = f x in  (*se usa comodin para que no chille*)
    Sys.time() -. t;;  

let rec fib2 n = 
  if n = 1 then 1, 0
  else 
    let f1, f2 = fib2 (n-1) in
    f1 + f2, f1;;

let rec fib2 n = function
    0 -> 0, 1
  | n ->  let f1, f2 = fib2 (n-1) in
          f1 + f2, f1;; 

let fib n = fst(fib2 n);;      

[1; 10; 100] (*int list*)

let l = [1;2;3;100;1000];;

[];;
(*'a list = []*)

1 :: l;; 

1::2::3::100::1000;;

let rec from_to m n = 
  if m > n then []
  else m::from_to(m+1) n;; (*no tiene recursion terminal*) 

let rec init n f = 
  if n = 0 then []
  else f 0 :: init(n-1)(function i -> f (i+1));; 

let hd = function
    h::_ -> hd
  | [] -> raise(Failure "hd")

Failure, Division_by_Zero, Invalid_argument constructores de tipo de datos exn   

let rec nth l n = 
  if l = [] then raise  (Failure "nth")
  else if n = 0 then List.hd l
  else nth (List.tl l) (n-1)      

let nth l n = if n >= 0 then nth l n
              else raise (Invalid_argument "nth")

let rec nth l n = match (l,n) with
     ([], _) -> raise (Failure "nth")
    |(h::_, 0) -> h
    |(_::t, n) -> nth t (n-1)           

let nth l n = if n > 0 then nth l n
    else raise (Invalid_argument "nth")

 let rec append l1 l2 = match l1 with
  [] -> l2
 |h:: -> h :: append t l2;;
 
 let rec compare_legths l2 l2 = match l1, l2 with
    [],[] -> 0
   |[], _ -> -1
   |_, [] -> 1
   | _::t1, _::t2 -> compare_legths t1 t2;;

 let rec aux i = function
    [] -> i
   |_::t -> aux (i+1) t;;

let length l = 
  let rec aux i = function
    [] -> i
    |_::t -> aux (i+1) 
in aux 0 l;;   

let fact n = 
  let rec aux i f =
    if i = n then f
    else aux (i+1)((i+1)*f)  

let fib n =
  let rec aux i f a =
    if i = n then f
    else aux (i+1) (f+a) f
  in aux 0 0 1;;       

let rec lmax = function
  [] -> raise(Failure "lmax");
 |h::[] -> h 
 |h::t -> max h (lmax t);;

let lmax = function
 [] -> raise(Failure "lmax");
 |h::t -> let rec aux m = function
            [] -> m
           |h::t-> aux (max m h) t
          in aux h t;;  

let rec rev_append l1 l2 = match l1 with
  [] -> l2
  |h::t -> rev_append t (h::l2);; 

let rev = rev_append l [];;

let append' l1 l2 = List.rev_append (List.rev_append l1) (l2);;

let rec fold_left f e = function
    [] -> e
   |h::t -> fold_left f (f e h) t;;

let sum l = fold_left (+) l 0;;

let last = function 
    [] -> raise(Failure "last")
   |h::t -> List.fold_left (function _ -> function y -> y) h t;;


let last = function 
    [] -> raise(Failure "last")
    |h::t -> List.fold_left (fun _ y -> y) h t;;

let length l = List.fold_left (fun s _ -> s + 1) 0 l;;

let rev l = List.fold_left (fun l' x -> x::l') [] l;;

let rec for_all p = function
      [] -> true
     |h::t -> p h && for_all p t;;
     
let for_all p l= List.fold_left (fun b x -> b && p x) true l;;

(*El primero es mejor porque la conjuncion si se encuentra
   un false ya da false*)
(*FUNCIONES QUE NO SON INTERESANTES CON FOLD_LEFT*)

let rec sorted = function
    [] | _::[] -> true
   |h1::h2::t -> h1 <= h2 && sorted (h2::t);;

let rec insert x = function
   [] -> [x] 
  |h::t -> if x <= h then x::h::t
           else h :: insert x t;;   
           
let rec isort = function 
  [] -> []
 |h::t -> insert h (isort t);;
 
 let insert' x l = 
	 let rec aux p1 p2 = match with 
			[] -> List.rev (x::p2)
		| h::t -> if x <= h then List.rev_append p2 (x::p1)
							else aux t (h::p2)
		in aux l [];;				 

let isort' l = 
		let rec aux l1 l2 = match l1 with 
			[] -> l2 
		|	h::t -> aux	t (insert' h l2)
	in aux l [];;

let isort' l = List.fold (fun t h -> insert' h t) [] l;;	

let rec insert ord x = function 
  [] -> [x]
 |h::t -> if ord x h then x::h::t 
          else h :: insert ord x t ;; 


let rec merge l1 l2 = match l1, l2 with 
    [], l | l, [] -> l       
   |h1::t1, h2::t2 -> if h1 <= h2 then h1 :: merge t1 l2 
                      else h2 :: merge l1 t2;;
                      
let rec divide l = match l with
   h1::h2::t -> let t1, t2 = divide t in
                h1::t1, h2::t2
  |_ -> l , [];;
  
let rec msort = function 
    [] -> []
   |[x] -> [x] 
   |l -> let l1, l2 = divide l in 
         merge (msort l1) (msort l2);;              

let rlist n = List.init n (fun _ -> Ramdon.int (2*n)) 

type 'a option = 
  None
 |Some of 'a;;   

type maybeAnInt = 
  NotAnInt
 |AnInt of int;;
 
let quo x y = match x, y with
  _, 0 -> NotAnInt
 |AnInt m, AnInt n -> AnInt(m/n)  
 |_ -> NotAnInt;;
 
 let foo = Foo;;

 let conj a b = match a,b with
  F,_ -> F 
 |_, F -> f
 |_ -> T;;
 
 let (&&&) = conj;;

 type t = T of int;;

 type tt = L of int | R of int;;

 type num = F of float | I of int;;

 type nat = Z | S of nat;;

 let suma x y = match x with 
  Z -> y
 |S n -> suma n (S y);;

 sum SSSSSZ SSSZ;;
 sum SSSSSSZ SSZ;;
 sum SSSSSSSZ SZ;;
 sum SSSSSSSSZ Z;; (*Caso base de la recursividad*)
 SSSSSSSSSZ;; 

let nat_of_int = function 
  0 -> Z 
  | n -> if n < 0 then raise (Invalid_argument "nat_of_int")
        else S (nat_of_int (n-1));;

let rec nat_of_int = function
  0 -> Z
 |n-> S (nat_of_int (n-1));;

let nat_of_int n = 
  if n<0 then raise (Invalid_argument "nat_of_int")
  else nat_of_int n;;

type 'a btree = 
  E
 |N of a' * a' btree * a' btree;;

N (2, E, E);;

let l x = N (x, E, E);; (*Nodo hoja uwu*)

let t6 = N (6, l 5, l 11);;
let t7 = N (7, l 2, t 6);;
let t9 = N (9, l 4, E);;
let t5 = N (5, E, t9);;
let t2 = N (2, t7, t5);;

let rec num_nodes  = function
  E -> 0
 |N(_,lb, rb) -> 1 + num_nodes lb + num_nodes rb;; 

let rec height = function
  E -> 0
 |N (_, lb, rb) -> 1 + max (height lb) (height rb);;

let preorder = function
  E -> []
 |N (x, lb, rb) -> (x::preorder lb) @ (preorder rb);;

 let rec leaves = function 
  E -> []
 |N(v,E,E) -> [v]
 |N(_,l,r) -> leaves l @ leaves r;; 

type 'a gtree = 
  GT of 'a * 'a gtree list;;

let h v = GT (v,[]);;

let t9 = GT (9, [h 4]);;
let t6 = GT (6, [h 5; h 11]);;
let t7 = GT (7, [h 2; h 10; t6]);;
let t5 = GT (5, [t9]);;
let t = GT(2, [t4;t5]);;

let rec nngt (GT(_,l)) = 
  List.foldleft (+) 1 (List.map nngt l);;

let rec nngt = function
  GT (_, []) -> 1
 |GT(v, h::t) -> nngt h + (GT(v, t));;
 
type 'a st_tree = 
  Leaf of 'a
 |Node of 'a * a' st_tree * a' st_tree;;
 
let rec mirror = function
  Leaf v -> Leaf v
 |Node (v,l,r) -> Node(v,mirror r, mirror l);; 

 let rec b_of_st = function
  Leaf v -> N(v, E, E)
 |Node (v, l, r) -> N(v, b_of_st l, b_of_st r);;

let rec st_of_b = function
  E -> raise(failure "st_of_b")
 |N(v,E,E) -> Leaf v
 |N(v,l,r) -> Node(v, st_of_b l, st_of_b r)   

output_char stdout 'X';;

let _ = print_char 'X' in print_char 'Y';;

print_char 'X'; print_char 'Y';;

let output_string c s = 
  let n = String.length s in 
  let rec loop i = 
    if i >= n then ()
    else (output_char c s.[i]; loop (i+1))
  in loop 0;;
  
let print_string s = output_string stdout s;;

let print_endline s = print_string(s ^ "\n"); flush stdout;;

let sal = open_out "pru";; (*crear archivo para escribir en el *) 


close_out sal;; (*cerrar archivo*)

open_in;;

let en = open_in "pru";;

input_char en;;

let rec output_string_list out = function
  [] -> ()
 |h::t -> output_string out (h^t); 
          output_string_list out t;;

let output_string_list out l = 
  List.iter (fun s -> output_string out (s^"\n")) l;;
  
let rec input_string_list input = 
  try let s = input_line input in
    s :: input_string_list input
  with End_of_file -> [];;
  
pos_in;; (*saber donde esta el puntero*)  
seek_in entrada 1;; (*cambiar el puntero*)
out_channel;;
output_value;;
input_value;;

Sys.command "clear"

let i = ref 0;;
(!);; 
(:=);;
i := !i + 1;;

let fact n = 
  let f = ref 1 in 
  for i = 0 to n do 
    f := !f * i
  done;
  !f;;  

 let fact n = 
  let f = ref 1 in 
  let i = ref 1 in
  while !f <= n do 
    f:= !f * !i;
    i:= !i + 1
  done;
  !f   
     
let n = ref 0;;

let turno () = 
  n := !n + 1;
  !n;;

let turno = 
  let n = ref 0 in 
  function () -> n:= !n + 1;
                !n;; 
                
let reset () =                 
  n:=0;;

let turno, reset =
  let n = ref 0 in 
  (fun () -> n:=!n + 1; !n),
  (fun () -> n:= 0 );;

module Counter () :
sig 
  val turno: unit -> int
  val reset: unit -> unit
end = 
struct
  let n = ref 0
  let turno () =
    n := !n + 1;
    !n
  let reset () = 
    n := 0
end;;     

module IntPair = 
struct 
  type t = int * int 
  let compare = Stdlib.compare
end


module IPSet = Set.Make (IntPair);;

let trees = List.init 50_000 (fun _ -> Random.int 500 + 1,
                              fun _ -> Random.int 500 +1)

let trees_S = IPSet.of_list trees;;

let trees_S = List.fold_left (fun s e -> IPSet.add e s) IPSet.empty trees;;

let to_find = List.init 5000 (fun _ -> Random.int 500 + 1,
                              fun _ -> Random.int 500 +1)

let r1 = List.filter (fun p -> List.mem p trees) to_find;;
let r2 = List.filter (fun p -> IPSet.mem p trees_s) to_find;;

let v = [|1;2;3|] (*ARRAY*)

Array.get v 1
v.(1)
Array.set v 2 1000
v.(2) <- 1000

let sprod v1 v2 = 
  if Array.length v1 <> Array.length v2
    then raise (Invalid_argument "sprod")
  else begin
    let p = ref 0.0 in
    for i = 0 to  Array.length v1 do
      p := !p +. v1.(i) . v2.(i)
    done;
    !p
  end
;;

let sprod v1 v2 =
  Array.fold_left (+.) 0.0 (Array.map2 (.) v1 v2);;

type persona = {nombre: string; edad: int};;
let pepe = {nombre = "Pepe"; edad = 57};;
{edad = 21; nombre "Maria"};;
pepe.edad;;

let mas_viejo p = 
  {nombre = p.nombre; edad = p.edad + 1};;

type persona = {nombre: string; mutable edad: int};;
maria.edad <- 32;;
let  envejece p = p.edad <- p.edad+1;;

type 'a ref = {mutable contents : 'a};;

let (!) v = v.contents;;
let(:=) v x = v.contents <- x;;
let ref x = {contets = x};;

type counter = {turno: unit -> int; reset: unit -> unit};;

let c1 = 
  let n = ref 0 in 
  {turno = (fun () -> n:= !n + 1; !n)
   reset = (fun () -> n:= 0)};;

let make_counter () =
  let n = ref 0 in 
  {turno = (fun () -> n:= !n + 1; !n)
   reset = (fun () -> n:= 0)};;

let rec par n = 
  n = 0 || impar (n-1)
and impar n = 
  n <> 0 && par (n-1);;
  
(*orientacion a objetos*)

let c = object
  val mutable n = 0
  method turno = 
    n <- n + 1;
    n
  method reset = 
    n <- 0
end      

class counter = object
  val mutable n = 0
  method turno = 
    n <- n + 1;
    n
  method reset = 
    n <- 0
end   

class counter_w_set = object 
inherit counter 
method set i = 
  n<-i
end;;  

let cs1 = new counter_w_set;;

let cs1' = (cs1:> counter) 